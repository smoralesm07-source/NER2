#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Taxonomía canónica compartida por todas las capas del pipeline.

Este módulo es la única fuente de verdad sobre:

1. Los TIPOS de entidad (compatibles con ``reconocedor_entidades.py`` v3.0.0).
2. La naturaleza jurídica derivada de cada tipo.
3. Las etiquetas en lenguaje natural que se le pasan a GLiNER (zero-shot) y su
   mapeo de vuelta a los tipos canónicos.
4. Los ROLES PROCESALES admisibles.

RESTRICCIÓN DE DISEÑO
---------------------
Los nombres de tipo NO se pueden cambiar sin romper ``entidades.html`` y el
consumo actual de ``datos.json``. Cualquier tipo nuevo debe agregarse, nunca
renombrarse.

RESTRICCIÓN LEGAL
-----------------
La taxonomía de roles procesales no admite ninguna etiqueta que impute
culpabilidad. "Formalizado", "imputado" y "acusado" describen un estado
procesal verificable en la fuente; no describen responsabilidad penal. No
existe ni debe existir un rol "CULPABLE", "LAVADOR" o equivalente. Ver
``ROLES_PROHIBIDOS``.
"""

from __future__ import annotations

import unicodedata

VERSION_TAXONOMIA = "1.0.0"


# ---------------------------------------------------------------------------
# 1. Tipos canónicos y naturaleza jurídica
#    (espejo exacto de NATURALEZA_POR_TIPO en reconocedor_entidades.py v3)
# ---------------------------------------------------------------------------

NATURALEZA_POR_TIPO: dict[str, str] = {
    "PERSONA": "PERSONA_NATURAL",
    "EMPRESA": "PERSONA_JURIDICA",
    "INSTITUCION_FINANCIERA": "PERSONA_JURIDICA",
    "ORGANISMO_PUBLICO": "PERSONA_JURIDICA",
    "ORGANIZACION": "PERSONA_JURIDICA",
    "ENTIDAD_SIN_FINES_DE_LUCRO": "PERSONA_JURIDICA",
    "TRIBUNAL": "PERSONA_JURIDICA",
    "LUGAR": "NO_APLICA",
    "MONTO": "NO_APLICA",
    "FECHA": "NO_APLICA",
    "RUT": "NO_APLICA",
    "CRIPTOACTIVO": "NO_APLICA",
    "OTRO": "INDETERMINADA",
}

TIPOS_CANONICOS: frozenset[str] = frozenset(NATURALEZA_POR_TIPO)

TIPOS_PERSONA_JURIDICA: frozenset[str] = frozenset(
    t for t, n in NATURALEZA_POR_TIPO.items() if n == "PERSONA_JURIDICA"
)

#: Tipos sobre los que el adjudicador LLM tiene autoridad para pronunciarse.
#: MONTO, FECHA y RUT se resuelven por regla determinista y no se someten al
#: modelo: pedirle que extraiga cifras es la vía más rápida a un número
#: inventado.
TIPOS_ADJUDICABLES: frozenset[str] = frozenset(
    {
        "PERSONA",
        "EMPRESA",
        "INSTITUCION_FINANCIERA",
        "ORGANISMO_PUBLICO",
        "ORGANIZACION",
        "ENTIDAD_SIN_FINES_DE_LUCRO",
        "TRIBUNAL",
        "LUGAR",
        "CRIPTOACTIVO",
        "OTRO",
    }
)


def naturaleza_de(tipo: str | None) -> str:
    """PERSONA_NATURAL / PERSONA_JURIDICA / NO_APLICA / INDETERMINADA."""
    return NATURALEZA_POR_TIPO.get(str(tipo or "").upper(), "INDETERMINADA")


# ---------------------------------------------------------------------------
# 2. Etiquetas GLiNER (zero-shot) -> tipo canónico
# ---------------------------------------------------------------------------
#
# GLiNER es zero-shot: las etiquetas se definen en tiempo de inferencia y el
# modelo las interpreta semánticamente. Esto implica dos cosas prácticas:
#
#   - Las etiquetas deben estar en español y ser descriptivas, no siglas.
#     "persona natural" funciona; "PN" no.
#   - Cada etiqueta adicional tiene costo de inferencia (el modelo evalúa cada
#     span contra cada etiqueta). Se mantienen 11, no 30.
#
# El mapeo es intencionalmente muchos-a-uno: varias etiquetas naturales
# colapsan al mismo tipo canónico para dar al modelo formas de expresar
# matices sin ampliar la taxonomía de salida.

ETIQUETAS_GLINER: dict[str, str] = {
    "persona": "PERSONA",
    "empresa o sociedad comercial": "EMPRESA",
    "banco o institución financiera": "INSTITUCION_FINANCIERA",
    "organismo público o servicio del Estado": "ORGANISMO_PUBLICO",
    "tribunal o corte de justicia": "TRIBUNAL",
    "fundación, corporación u organización sin fines de lucro": "ENTIDAD_SIN_FINES_DE_LUCRO",
    "asociación gremial, sindicato o partido político": "ORGANIZACION",
    "comuna, ciudad, región o país": "LUGAR",
    "criptoactivo o exchange de criptomonedas": "CRIPTOACTIVO",
    "monto de dinero": "MONTO",
    "RUT o rol único tributario": "RUT",
}

#: Lista que se le entrega literalmente a ``predict_entities(labels=...)``.
ETIQUETAS_GLINER_LISTA: list[str] = list(ETIQUETAS_GLINER)


def tipo_desde_etiqueta_gliner(etiqueta: str) -> str:
    """Mapea una etiqueta devuelta por GLiNER al tipo canónico.

    GLiNER puede devolver la etiqueta con variaciones de mayúsculas. La
    comparación se hace normalizada; si no hay coincidencia, se devuelve OTRO
    en vez de fallar, porque una etiqueta desconocida es un candidato válido
    que el adjudicador puede clasificar después.
    """
    directo = ETIQUETAS_GLINER.get(etiqueta)
    if directo:
        return directo
    clave = _plegar(etiqueta)
    for original, tipo in ETIQUETAS_GLINER.items():
        if _plegar(original) == clave:
            return tipo
    return "OTRO"


# ---------------------------------------------------------------------------
# 3. Roles procesales
# ---------------------------------------------------------------------------
#
# El valor analítico de una entidad en prensa no está en la entidad sino en su
# estado respecto de un procedimiento. "Juan Pérez" no sirve; "Juan Pérez,
# formalizado por lavado de activos el 3 de marzo" sí.
#
# El orden de la tupla es de menor a mayor gravedad procesal y se usa para
# priorizar la cola de revisión, no para inferir culpabilidad.

ROLES_PROCESALES: tuple[str, ...] = (
    "SIN_ROL",                        # aparece en el texto sin vínculo procesal
    "MENCIONADO",                     # citado en el contexto del caso, sin calidad
    "VICTIMA",
    "TESTIGO",
    "DENUNCIANTE",
    "QUERELLANTE",
    "DEFENSA",                        # abogado o defensor de un tercero
    "FISCALIZADOR",                   # organismo que investiga o fiscaliza
    "REPORTANTE",                     # sujeto obligado que reportó
    "DENUNCIADO",
    "INVESTIGADO",
    "IMPUTADO",
    "FORMALIZADO",
    "ACUSADO",
    "SANCIONADO_ADMINISTRATIVAMENTE",
    "ABSUELTO",
    "SOBRESEIDO",
    "CONDENADO",
)

ROLES_PROCESALES_SET: frozenset[str] = frozenset(ROLES_PROCESALES)

#: Roles que implican una atribución de responsabilidad no establecida por una
#: resolución.
#:
#: Si el modelo devuelve uno de estos, el rol se **anula** (SIN_ROL) y la
#: entidad se marca para validación humana. No se suaviza a INVESTIGADO: hacerlo
#: seguiría afirmando una calidad procesal cuya única fuente es un juicio del
#: modelo que ya se descartó por inadmisible. Si el artículo efectivamente
#: sostiene que la persona está siendo investigada, el analista lo verá en la
#: evidencia, que sí se conserva.
#:
#: Es una barrera de seguridad, no una expectativa: el prompt ya prohíbe estos
#: valores.
ROLES_PROHIBIDOS: frozenset[str] = frozenset(
    {
        "CULPABLE",
        "RESPONSABLE",
        "LAVADOR",
        "DELINCUENTE",
        "CRIMINAL",
        "AUTOR",
        "PARTICIPE",
        "NARCOTRAFICANTE",
        "TESTAFERRO",
    }
)

#: Roles que implican riesgo reputacional relevante para adverse media
#: screening. Determinan la prioridad en la cola del analista.
ROLES_ADVERSOS: frozenset[str] = frozenset(
    {
        "DENUNCIADO",
        "INVESTIGADO",
        "IMPUTADO",
        "FORMALIZADO",
        "ACUSADO",
        "SANCIONADO_ADMINISTRATIVAMENTE",
        "CONDENADO",
    }
)


def normalizar_rol(rol: str | None) -> tuple[str, bool]:
    """Devuelve ``(rol_canonico, fue_degradado)``.

    La documentación de structured outputs advierte que la capitalización de
    los valores de enum no está garantizada, de modo que la comparación se hace
    insensible a mayúsculas y tildes. Un rol prohibido se anula y se señala para
    que la capa de fusión marque ``requiere_validacion``.
    """
    crudo = str(rol or "").strip()
    if not crudo:
        return "SIN_ROL", False

    clave = _plegar(crudo).replace(" ", "_").replace("-", "_").upper()

    if clave in ROLES_PROHIBIDOS:
        return "SIN_ROL", True

    for canonico in ROLES_PROCESALES:
        if _plegar(canonico) == _plegar(clave):
            return canonico, False

    return "SIN_ROL", True


def normalizar_tipo(tipo: str | None) -> tuple[str, bool]:
    """Devuelve ``(tipo_canonico, fue_degradado)`` con la misma tolerancia."""
    crudo = str(tipo or "").strip()
    if not crudo:
        return "OTRO", True

    clave = _plegar(crudo).replace(" ", "_").replace("-", "_").upper()
    for canonico in TIPOS_CANONICOS:
        if _plegar(canonico) == _plegar(clave):
            return canonico, False
    return "OTRO", True


# ---------------------------------------------------------------------------
# 4. Utilidad de normalización
# ---------------------------------------------------------------------------


def _plegar(texto: str) -> str:
    """Minúsculas sin tildes ni diacríticos, para comparación robusta.

    El bug de v2.1.1 en que 'fundacion' nunca coincidía con 'Fundación' salía
    de no tener una función así en un solo lugar.
    """
    descompuesto = unicodedata.normalize("NFD", str(texto or ""))
    sin_marcas = "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", sin_marcas).casefold().strip()


plegar = _plegar  # alias público


# ---------------------------------------------------------------------------
# 5. Procedencia y precedencia
# ---------------------------------------------------------------------------
#
# Cada entidad arrastra la capa que la produjo. Ante conflicto de clasificación
# gana la fuente con mayor precedencia. La regla determinista gana sobre el
# LLM: si la cadena termina en "Ltda." es persona jurídica aunque el modelo
# argumente lo contrario.

PRECEDENCIA_FUENTE: dict[str, int] = {
    "REGLA": 300,      # L1 - sufijo societario, RUT válido, gazetteer
    "LLM": 200,        # L3 - adjudicación semántica
    "GLINER": 100,     # L2 - candidato zero-shot
    "SPACY": 90,       # L2 - respaldo si GLiNER no está disponible
    "DESCONOCIDA": 0,
}


def precedencia(fuente: str | None) -> int:
    return PRECEDENCIA_FUENTE.get(str(fuente or "").upper(), 0)
