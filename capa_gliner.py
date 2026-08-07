#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capa L2 — Generación de candidatos con GLiNER (zero-shot).

Por qué GLiNER reemplaza a ``es_core_news_sm``
----------------------------------------------
El modelo pequeño de spaCy está entrenado sobre CoNLL/AnCora y solo sabe
cuatro clases: PER, ORG, LOC, MISC. Para producir "ORGANISMO_PUBLICO" o
"INSTITUCION_FINANCIERA" hay que post-procesar ORG con reglas, que es
exactamente donde se acumulan los errores. Además, su confusión PER↔ORG está
documentada precisamente en el caso que más importa aquí: razones sociales que
contienen antropónimos.

GLiNER resuelve el problema de raíz porque las etiquetas se definen en tiempo
de inferencia. Se le entrega la taxonomía UAF en español y clasifica contra
ella directamente, sin reentrenar y sin corpus anotado.

Restricción operativa: chunking obligatorio
-------------------------------------------
La configuración por defecto de GLiNER es ``max_len = 384`` tokens. Un artículo
de prensa chileno promedia entre 500 y 1.500 tokens, de modo que pasarle el
texto completo trunca silenciosamente todo lo que exceda el límite: se pierden
entidades sin ningún error visible. Este módulo segmenta el texto respetando
límites de párrafo y oración, ejecuta la inferencia por segmento y **remapea
los offsets al texto original**, que es lo que permite que la capa de
validación de spans funcione más adelante.

Los segmentos se solapan para no partir una entidad en el corte. El solape
genera duplicados, que se deduplican por span.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from taxonomia_uaf import (
    ETIQUETAS_GLINER_LISTA,
    tipo_desde_etiqueta_gliner,
)

VERSION_CAPA_GLINER = "1.0.0"

#: Modelo por defecto. La variante multilingüe rinde mejor en español que la
#: base en inglés y pesa ~500 MB, ejecutable en CPU.
MODELO_POR_DEFECTO = os.environ.get("GLINER_MODELO", "urchade/gliner_multi-v2.1")

#: Presupuesto de caracteres por segmento. 384 tokens en español rinden entre
#: 1.100 y 1.600 caracteres según la densidad de nombres propios; 1.100 deja
#: margen para el prefijo de etiquetas, que también consume tokens del límite.
CARACTERES_POR_SEGMENTO = 1100

#: Solape entre segmentos consecutivos, en caracteres. Debe superar el largo
#: de la entidad más larga esperada ("Juzgado de Garantía de Puente Alto").
SOLAPE = 180

UMBRAL_POR_DEFECTO = 0.45


_modelo_cargado: Any = None
_error_carga: str = ""


@dataclass
class CandidatoGliner:
    texto: str
    inicio: int
    fin: int
    tipo: str
    etiqueta_original: str
    score: float


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------


def cargar_modelo(modelo: str = MODELO_POR_DEFECTO) -> Any:
    """Carga perezosa y única del modelo. Devuelve ``None`` si no se puede.

    Cargar GLiNER cuesta entre 10 y 40 segundos en CPU. Se hace una sola vez
    por proceso: en GitHub Actions eso significa una vez por job, no una vez
    por artículo.
    """
    global _modelo_cargado, _error_carga

    if _modelo_cargado is not None:
        return _modelo_cargado
    if _error_carga:
        return None

    try:
        from gliner import GLiNER  # type: ignore
    except Exception as exc:
        _error_carga = (
            f"GLiNER no está instalado ({exc}). Instale 'gliner' y 'torch' o "
            "el pipeline operará con la capa de reglas y el adjudicador LLM."
        )
        return None

    try:
        _modelo_cargado = GLiNER.from_pretrained(modelo, map_location="cpu")
        try:
            _modelo_cargado.eval()
        except Exception:
            pass
    except Exception as exc:
        _error_carga = f"No se pudo cargar el modelo '{modelo}': {exc}"
        return None

    return _modelo_cargado


def disponible() -> bool:
    return cargar_modelo() is not None


def error_carga() -> str:
    return _error_carga


# ---------------------------------------------------------------------------
# Segmentación con offsets absolutos
# ---------------------------------------------------------------------------

_RE_CORTE = re.compile(r"(?<=[.!?])\s+(?=[¿¡\"“'(A-ZÁÉÍÓÚÑÜ0-9])|\n+")


def segmentar(
    texto: str,
    maximo: int = CARACTERES_POR_SEGMENTO,
    solape: int = SOLAPE,
) -> list[tuple[str, int]]:
    """Divide el texto en ``(fragmento, offset_absoluto)``.

    El corte se hace en frontera de oración o párrafo cuando existe una dentro
    del presupuesto; si una sola oración excede el presupuesto (frecuente en
    prensa judicial, con oraciones de 300 palabras), se corta por espacio en
    blanco, y solo en último término a la fuerza.
    """
    if not texto:
        return []
    if len(texto) <= maximo:
        return [(texto, 0)]

    # Posiciones candidatas de corte, en orden.
    cortes = [m.end() for m in _RE_CORTE.finditer(texto)]

    segmentos: list[tuple[str, int]] = []
    inicio = 0
    while inicio < len(texto):
        limite = inicio + maximo
        if limite >= len(texto):
            segmentos.append((texto[inicio:], inicio))
            break

        candidatos = [c for c in cortes if inicio + (maximo // 3) < c <= limite]
        if candidatos:
            corte = candidatos[-1]
        else:
            espacio = texto.rfind(" ", inicio + (maximo // 3), limite)
            corte = espacio if espacio > inicio else limite

        segmentos.append((texto[inicio:corte], inicio))
        siguiente = corte - solape
        inicio = siguiente if siguiente > inicio else corte

    return segmentos


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------


def detectar(
    texto: str,
    umbral: float = UMBRAL_POR_DEFECTO,
    etiquetas: list[str] | None = None,
    modelo: str = MODELO_POR_DEFECTO,
) -> list[CandidatoGliner]:
    """Ejecuta GLiNER sobre el texto completo y devuelve candidatos.

    Los offsets devueltos son absolutos respecto de ``texto``, no del segmento.
    """
    motor = cargar_modelo(modelo)
    if motor is None or not texto:
        return []

    labels = etiquetas or ETIQUETAS_GLINER_LISTA
    crudos: list[CandidatoGliner] = []

    for fragmento, offset in segmentar(texto):
        try:
            predicciones = motor.predict_entities(
                fragmento,
                labels,
                threshold=umbral,
                flat_ner=True,
            )
        except Exception:
            continue

        for pred in predicciones:
            inicio_abs = offset + int(pred["start"])
            fin_abs = offset + int(pred["end"])
            superficie = texto[inicio_abs:fin_abs]

            # Verificación de remapeo: si el span absoluto no reproduce lo que
            # el modelo dijo haber encontrado, el offset está corrido y el
            # candidato se descarta antes de contaminar las capas siguientes.
            if superficie != pred.get("text", superficie):
                continue

            crudos.append(
                CandidatoGliner(
                    texto=superficie,
                    inicio=inicio_abs,
                    fin=fin_abs,
                    tipo=tipo_desde_etiqueta_gliner(str(pred.get("label", ""))),
                    etiqueta_original=str(pred.get("label", "")),
                    score=float(pred.get("score", 0.0)),
                )
            )

    return _deduplicar(crudos)


def _deduplicar(candidatos: list[CandidatoGliner]) -> list[CandidatoGliner]:
    """Elimina duplicados del solape y resuelve spans anidados.

    Dos predicciones del mismo span (una por cada segmento que lo contiene) se
    colapsan quedándose con la de mayor score. Un span contenido en otro se
    descarta en favor del más largo, salvo que sean de tipos distintos, en cuyo
    caso ambos sobreviven y el adjudicador decide.
    """
    por_span: dict[tuple[int, int], CandidatoGliner] = {}
    for cand in candidatos:
        clave = (cand.inicio, cand.fin)
        previo = por_span.get(clave)
        if previo is None or cand.score > previo.score:
            por_span[clave] = cand

    unicos = sorted(por_span.values(), key=lambda c: (c.inicio, -(c.fin - c.inicio)))

    salida: list[CandidatoGliner] = []
    for cand in unicos:
        contenido = False
        for otro in salida:
            mismo_tipo = otro.tipo == cand.tipo
            dentro = otro.inicio <= cand.inicio and cand.fin <= otro.fin
            if dentro and mismo_tipo:
                contenido = True
                break
        if not contenido:
            salida.append(cand)
    return salida


# ---------------------------------------------------------------------------
# Respaldo con spaCy
# ---------------------------------------------------------------------------


def detectar_con_respaldo_spacy(texto: str) -> list[CandidatoGliner]:
    """Usa spaCy si GLiNER no está disponible. Marca los tipos como gruesos.

    El mapeo PER→PERSONA, ORG→OTRO es deliberado: spaCy no puede distinguir
    empresa de organismo público, y hacer que ORG caiga en EMPRESA introduce un
    sesgo sistemático. ORG→OTRO deja la decisión al adjudicador.
    """
    try:
        import spacy  # type: ignore
    except Exception:
        return []

    for nombre in ("es_core_news_lg", "es_core_news_md", "es_core_news_sm"):
        try:
            nlp = spacy.load(nombre)
            break
        except Exception:
            continue
    else:
        return []

    mapa = {"PER": "PERSONA", "LOC": "LUGAR", "ORG": "OTRO", "MISC": "OTRO"}
    doc = nlp(texto)
    return [
        CandidatoGliner(
            texto=ent.text,
            inicio=ent.start_char,
            fin=ent.end_char,
            tipo=mapa.get(ent.label_, "OTRO"),
            etiqueta_original=f"spacy:{ent.label_}",
            score=0.5,
        )
        for ent in doc.ents
    ]


def diagnostico() -> dict[str, Any]:
    return {
        "version": VERSION_CAPA_GLINER,
        "modelo": MODELO_POR_DEFECTO,
        "disponible": _modelo_cargado is not None,
        "error": _error_carga,
        "n_etiquetas": len(ETIQUETAS_GLINER_LISTA),
        "caracteres_por_segmento": CARACTERES_POR_SEGMENTO,
        "solape": SOLAPE,
    }
