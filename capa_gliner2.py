#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capa L2 alternativa — GLiNER2 (fastino-ai).

Expone exactamente la misma interfaz que ``capa_gliner.py`` (``detectar()``
devolviendo ``CandidatoGliner``), de modo que L3 y L4 no cambian. Se elige el
motor con la variable de entorno ``GLINER_BACKEND=v1|v2``.

Cuatro diferencias respecto de GLiNER v1 que hay que manejar
------------------------------------------------------------

**1. El modelo del README es solo inglés.**
La documentación oficial usa ``fastino/gliner2-base-v1``, que en Hugging Face
está etiquetado ``English``. Para prensa chilena hay que usar
``fastino/gliner2-multi-v1``, que no aparece en el README y sí en la colección
del autor. Usar el modelo del README sobre texto en español es la forma más
rápida de concluir, equivocadamente, que GLiNER2 no sirve.

**2. Los offsets NO vienen por defecto.**
``include_spans`` es ``False`` salvo que se pida. Sin offsets, la barrera de
anclaje y la fusión por solapamiento de ``fusion_entidades.py`` no pueden
operar: todo el diseño del pipeline depende de poder señalar la posición exacta
en el texto. Aquí se fuerza a ``True`` y no es configurable.

**3. La salida viene agrupada por etiqueta, no como lista plana.**
GLiNER v1 devuelve ``[{start, end, text, label, score}, ...]``. GLiNER2
devuelve ``{"entities": {"persona": [{...}, {...}], "empresa": [...]}}``. Este
módulo lo aplana.

**4. Las etiquetas admiten descripción.**
Esta es la ventaja real sobre v1: en vez de la cadena suelta ``"persona"`` se
puede pasar ``{"persona": "descripción de qué cuenta y qué no"}``. Para la
taxonomía UAF importa mucho, porque el caso difícil —una razón social que
contiene un apellido— se puede describir explícitamente en vez de esperar que
el modelo lo infiera.

Lo que GLiNER2 además ofrece y aquí no se usa
---------------------------------------------
Trae clasificación de texto y extracción de relaciones en la misma pasada. No
se cablean por una razón concreta: sus relaciones no vienen con la frase del
artículo que las sustenta, y sin evidencia verificable una relación no es
utilizable para análisis AML. Las relaciones siguen saliendo de L3, donde cada
una se descarta si su frase de sustento no existe en el texto.
"""

from __future__ import annotations

import os
from typing import Any

from capa_gliner import CandidatoGliner, segmentar
from taxonomia_uaf import plegar

VERSION_CAPA_GLINER2 = "1.0.0"

#: Modelo multilingüe. NO usar 'fastino/gliner2-base-v1' (solo inglés) ni
#: 'gliner2-large-v1' (también inglés) para prensa en español.
MODELO_POR_DEFECTO = os.environ.get("GLINER2_MODELO", "fastino/gliner2-multi-v1")

UMBRAL_POR_DEFECTO = 0.45

#: GLiNER2 no trunca por defecto (``max_len=None``), pero el encoder tiene un
#: límite posicional propio y la documentación advierte que los tokens que
#: excedan ``max_len`` se descartan en silencio. Se mantiene la segmentación
#: como seguro: los offsets se remapean igual y el solape se deduplica, así que
#: no cuesta nada correctitud y protege contra pérdida silenciosa de entidades
#: en la cola de artículos largos.
CARACTERES_POR_SEGMENTO = 1600
SOLAPE = 180


# ---------------------------------------------------------------------------
# Esquema con descripciones
# ---------------------------------------------------------------------------
#
# La descripción es lo que hace a este motor superior a v1 para el caso UAF.
# Cada una está redactada contra el error concreto que se quiere evitar.

ESQUEMA_UAF: dict[str, str] = {
    "persona": (
        "Nombre propio de un ser humano individual: nombre y apellidos. "
        "NO incluye nombres de empresas que contengan apellidos, ni nombres de "
        "comunas que provengan de personajes históricos."
    ),
    "empresa": (
        "Razón social de una sociedad comercial chilena, habitualmente con "
        "sufijo SpA, S.A., Ltda., Limitada, E.I.R.L. o «y Compañía». Es empresa "
        "aunque el nombre contenga apellidos de personas."
    ),
    "institucion_financiera": (
        "Banco, cooperativa de ahorro y crédito, corredora de bolsa, compañía "
        "de seguros, administradora general de fondos (AGF), AFP, casa de "
        "cambio o exchange de criptomonedas."
    ),
    "organismo_publico": (
        "Órgano del Estado de Chile: ministerio, subsecretaría, servicio "
        "público, superintendencia, fiscalía, municipalidad, Contraloría, "
        "Comisión para el Mercado Financiero, Servicio de Impuestos Internos, "
        "Unidad de Análisis Financiero, PDI, Carabineros."
    ),
    "tribunal": (
        "Órgano jurisdiccional: Juzgado de Garantía, Tribunal Oral en lo Penal, "
        "Corte de Apelaciones, Corte Suprema, Tribunal Constitucional."
    ),
    "sin_fines_de_lucro": (
        "Fundación, corporación, universidad u organización no gubernamental."
    ),
    "organizacion": (
        "Asociación gremial, sindicato, confederación, federación o partido "
        "político."
    ),
    "lugar": (
        "Topónimo: comuna, ciudad, provincia, región o país. Muchas comunas "
        "chilenas llevan nombre de persona (Pedro Aguirre Cerda, San Ramón, "
        "Padre Hurtado); en contexto territorial son lugares, no personas."
    ),
    "criptoactivo": "Criptomoneda o plataforma de intercambio de criptoactivos.",
    "monto": "Cantidad de dinero con su unidad o moneda.",
    "rut": "Rol Único Tributario chileno, con formato NN.NNN.NNN-D.",
}

#: Etiqueta de GLiNER2 -> tipo canónico del Monitor UAF.
MAPA_TIPO: dict[str, str] = {
    "persona": "PERSONA",
    "empresa": "EMPRESA",
    "institucion_financiera": "INSTITUCION_FINANCIERA",
    "organismo_publico": "ORGANISMO_PUBLICO",
    "tribunal": "TRIBUNAL",
    "sin_fines_de_lucro": "ENTIDAD_SIN_FINES_DE_LUCRO",
    "organizacion": "ORGANIZACION",
    "lugar": "LUGAR",
    "criptoactivo": "CRIPTOACTIVO",
    "monto": "MONTO",
    "rut": "RUT",
}


def tipo_desde_etiqueta(etiqueta: str) -> str:
    directo = MAPA_TIPO.get(etiqueta)
    if directo:
        return directo
    clave = plegar(etiqueta)
    for original, tipo in MAPA_TIPO.items():
        if plegar(original) == clave:
            return tipo
    return "OTRO"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

_modelo: Any = None
_error_carga: str = ""


def cargar_modelo(modelo: str = MODELO_POR_DEFECTO) -> Any:
    global _modelo, _error_carga

    if _modelo is not None:
        return _modelo
    if _error_carga:
        return None

    try:
        from gliner2 import GLiNER2  # type: ignore
    except Exception as exc:
        _error_carga = (
            f"gliner2 no está instalado ({exc}). Instale 'gliner2' o use "
            "GLINER_BACKEND=v1."
        )
        return None

    try:
        _modelo = GLiNER2.from_pretrained(modelo)
    except Exception as exc:
        _error_carga = f"No se pudo cargar el modelo '{modelo}': {exc}"
        return None

    if "multi" not in modelo.lower():
        _error_carga = ""  # no impide operar, pero se advierte
        print(
            f"AVISO: '{modelo}' no es el modelo multilingüe. Los modelos "
            "gliner2-base-v1 y gliner2-large-v1 están entrenados en inglés y "
            "rinden mal en prensa chilena. Use 'fastino/gliner2-multi-v1'.",
            flush=True,
        )

    return _modelo


def disponible() -> bool:
    return cargar_modelo() is not None


def error_carga() -> str:
    return _error_carga


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------


def detectar(
    texto: str,
    umbral: float = UMBRAL_POR_DEFECTO,
    etiquetas: dict[str, str] | None = None,
    modelo: str = MODELO_POR_DEFECTO,
) -> list[CandidatoGliner]:
    """Ejecuta GLiNER2 y devuelve candidatos con offsets absolutos."""
    motor = cargar_modelo(modelo)
    if motor is None or not texto:
        return []

    esquema = etiquetas or ESQUEMA_UAF
    crudos: list[CandidatoGliner] = []

    for fragmento, offset in segmentar(texto, CARACTERES_POR_SEGMENTO, SOLAPE):
        try:
            salida = motor.extract_entities(
                fragmento,
                esquema,
                threshold=umbral,
                include_spans=True,       # sin esto no hay offsets y el
                include_confidence=True,  # pipeline completo deja de funcionar
            )
        except Exception:
            continue

        # {"entities": {"persona": [{text,start,end,confidence}, ...], ...}}
        agrupadas = (salida or {}).get("entities", {}) or {}
        for etiqueta, elementos in agrupadas.items():
            if isinstance(elementos, dict):
                elementos = [elementos]
            if not isinstance(elementos, list):
                continue

            for item in elementos:
                if not isinstance(item, dict):
                    continue
                inicio_rel = item.get("start")
                fin_rel = item.get("end")
                if inicio_rel is None or fin_rel is None:
                    continue

                inicio_abs = offset + int(inicio_rel)
                fin_abs = offset + int(fin_rel)
                superficie = texto[inicio_abs:fin_abs]

                # Verificación de remapeo: si el offset absoluto no reproduce
                # lo que el modelo dijo haber encontrado, está corrido y el
                # candidato se descarta antes de contaminar las capas
                # siguientes.
                declarado = str(item.get("text", superficie))
                if superficie != declarado:
                    continue

                crudos.append(
                    CandidatoGliner(
                        texto=superficie,
                        inicio=inicio_abs,
                        fin=fin_abs,
                        tipo=tipo_desde_etiqueta(str(etiqueta)),
                        etiqueta_original=str(etiqueta),
                        score=float(item.get("confidence", 0.0) or 0.0),
                    )
                )

    return _deduplicar(crudos)


def _deduplicar(candidatos: list[CandidatoGliner]) -> list[CandidatoGliner]:
    """Colapsa duplicados del solape y spans anidados del mismo tipo."""
    por_span: dict[tuple[int, int, str], CandidatoGliner] = {}
    for cand in candidatos:
        clave = (cand.inicio, cand.fin, cand.tipo)
        previo = por_span.get(clave)
        if previo is None or cand.score > previo.score:
            por_span[clave] = cand

    unicos = sorted(por_span.values(), key=lambda c: (c.inicio, -(c.fin - c.inicio)))

    salida: list[CandidatoGliner] = []
    for cand in unicos:
        contenido = any(
            otro.tipo == cand.tipo and otro.inicio <= cand.inicio and cand.fin <= otro.fin
            for otro in salida
        )
        if not contenido:
            salida.append(cand)
    return salida


def diagnostico() -> dict[str, Any]:
    return {
        "version": VERSION_CAPA_GLINER2,
        "backend": "gliner2",
        "modelo": MODELO_POR_DEFECTO,
        "multilingue": "multi" in MODELO_POR_DEFECTO.lower(),
        "disponible": _modelo is not None,
        "error": _error_carga,
        "n_etiquetas": len(ESQUEMA_UAF),
        "caracteres_por_segmento": CARACTERES_POR_SEGMENTO,
        "solape": SOLAPE,
    }
