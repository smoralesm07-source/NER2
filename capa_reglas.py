#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capa L1 — Reconocimiento determinista de alta precisión.

Esta capa solo produce entidades sobre las que existe evidencia formal en el
texto: un sufijo societario, un encabezado institucional, un RUT con dígito
verificador válido. No infiere. Su precisión esperada es cercana al 100% y su
recall es deliberadamente bajo.

Ese perfil es lo que la hace útil: sus salidas tienen la precedencia más alta
en la fusión (``PRECEDENCIA_FUENTE["REGLA"] = 300``) y actúan como ancla contra
la que se contrasta lo que proponen GLiNER y el adjudicador LLM.

Acoplamiento con el Monitor UAF
-------------------------------
Si ``reconocedor_entidades.py`` v3 está presente en el path, esta capa lo
importa y usa su gazetteer geográfico y sus léxicos. Si no está, opera con su
propio conjunto reducido de reglas. En ningún caso falla por ausencia.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from taxonomia_uaf import plegar

VERSION_REGLAS = "1.0.0"


# Acoplamiento opcional con el reconocedor v3 del Monitor UAF.
try:
    import reconocedor_entidades as _v3  # type: ignore

    RECONOCEDOR_V3_DISPONIBLE = True
    VERSION_V3 = getattr(_v3, "VERSION_RECONOCEDOR", "desconocida")
except Exception:  # pragma: no cover
    _v3 = None
    RECONOCEDOR_V3_DISPONIBLE = False
    VERSION_V3 = ""

try:
    import geografia_cl as _geo  # type: ignore

    GEOGRAFIA_DISPONIBLE = True
except Exception:  # pragma: no cover
    _geo = None
    GEOGRAFIA_DISPONIBLE = False


# ---------------------------------------------------------------------------
# 1. RUT
# ---------------------------------------------------------------------------

_RE_RUT = re.compile(
    r"\b(\d{1,2}\.?\d{3}\.?\d{3})\s*[-‐‑–]\s*([\dkK])\b"
)


def digito_verificador(cuerpo: str) -> str:
    """Dígito verificador por módulo 11."""
    suma = 0
    multiplicador = 2
    for digito in reversed(cuerpo):
        suma += int(digito) * multiplicador
        multiplicador = 2 if multiplicador == 7 else multiplicador + 1
    resto = 11 - (suma % 11)
    if resto == 11:
        return "0"
    if resto == 10:
        return "K"
    return str(resto)


def rut_valido(rut: str) -> bool:
    coincidencia = _RE_RUT.search(str(rut or ""))
    if not coincidencia:
        return False
    cuerpo = coincidencia.group(1).replace(".", "")
    return digito_verificador(cuerpo) == coincidencia.group(2).upper()


def formatear_rut(cuerpo: str, dv: str) -> str:
    limpio = cuerpo.replace(".", "")
    partes: list[str] = []
    while len(limpio) > 3:
        partes.insert(0, limpio[-3:])
        limpio = limpio[:-3]
    partes.insert(0, limpio)
    return f"{'.'.join(partes)}-{dv.upper()}"


#: Los RUT de persona jurídica en Chile parten en 50.000.000; bajo ese umbral
#: corresponden a personas naturales. Es una señal fuerte de naturaleza que no
#: requiere ninguna inferencia semántica.
UMBRAL_RUT_JURIDICA = 50_000_000


# ---------------------------------------------------------------------------
# 2. Sufijos societarios
# ---------------------------------------------------------------------------
#
# El sufijo es determinante: si una cadena termina en "Ltda." es una persona
# jurídica, aunque el nombre completo sea "Juan Pérez y Compañía Ltda." y todo
# modelo de NER quiera etiquetarla como persona.

_SUFIJOS: tuple[tuple[str, str], ...] = (
    (r"S\.?\s?p\.?\s?A\.?", "EMPRESA"),
    (r"S\.?\s?A\.?", "EMPRESA"),
    (r"S\.?\s?A\.?\s?C\.?\s?I\.?", "EMPRESA"),
    (r"Ltda\.?", "EMPRESA"),
    (r"Limitada", "EMPRESA"),
    (r"E\.?\s?I\.?\s?R\.?\s?L\.?", "EMPRESA"),
    (r"S\.?\s?G\.?\s?R\.?", "EMPRESA"),
    (r"y\s+(?:Cía|Cia|Compañía|Compania)\.?", "EMPRESA"),
    (r"S\.?\s?A\.?\s?Administradora\s+General\s+de\s+Fondos", "INSTITUCION_FINANCIERA"),
    (r"A\.?\s?G\.?\s?F\.?", "INSTITUCION_FINANCIERA"),
    (r"A\.?\s?F\.?\s?P\.?", "INSTITUCION_FINANCIERA"),
    (r"A\.?\s?G\.?", "ORGANIZACION"),
)

#: Un nombre propio en Chile puede tener hasta 5 o 6 palabras antes del sufijo
#: ("Inversiones y Asesorías del Pacífico Sur SpA"). Se limita a 6 para no
#: engullir la oración anterior.
_RE_SUFIJO = re.compile(
    r"\b((?:[A-ZÁÉÍÓÚÑÜ][\wÁÉÍÓÚÑÜáéíóúñü'’\-]*(?:\s+(?:de|del|la|las|los|y|e)\s+|\s+)){0,6}"
    r"[A-ZÁÉÍÓÚÑÜ][\wÁÉÍÓÚÑÜáéíóúñü'’\-]*)\s+"
    r"(" + "|".join(p for p, _ in _SUFIJOS) + r")(?=[\s,.;:)\]]|$)"
)

_MAPA_SUFIJO_TIPO: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"^{patron}$", re.IGNORECASE), tipo) for patron, tipo in _SUFIJOS
]


# ---------------------------------------------------------------------------
# 3. Encabezados institucionales
# ---------------------------------------------------------------------------
#
# El encabezado ("Ministerio de", "Juzgado de Garantía de") determina el tipo
# con la misma fuerza que un sufijo, y en dirección contraria: aparece al
# inicio del nombre.

_ENCABEZADOS: tuple[tuple[str, str], ...] = (
    (r"Ministerio(?:\s+P[úu]blico)?", "ORGANISMO_PUBLICO"),
    (r"Subsecretar[íi]a", "ORGANISMO_PUBLICO"),
    (r"Superintendencia", "ORGANISMO_PUBLICO"),
    (r"Servicio(?:\s+Nacional)?", "ORGANISMO_PUBLICO"),
    (r"Direcci[óo]n(?:\s+Nacional|\s+General)?", "ORGANISMO_PUBLICO"),
    (r"Contralor[íi]a(?:\s+General)?", "ORGANISMO_PUBLICO"),
    (r"Comisi[óo]n(?:\s+para\s+el\s+Mercado\s+Financiero|\s+Nacional)?", "ORGANISMO_PUBLICO"),
    (r"Unidad\s+de\s+An[áa]lisis\s+Financiero", "ORGANISMO_PUBLICO"),
    (r"Municipalidad", "ORGANISMO_PUBLICO"),
    (r"Gobierno\s+Regional", "ORGANISMO_PUBLICO"),
    (r"Instituto(?:\s+Nacional)?", "ORGANISMO_PUBLICO"),
    (r"Fiscal[íi]a(?:\s+Regional|\s+Nacional)?", "ORGANISMO_PUBLICO"),
    (r"Polic[íi]a\s+de\s+Investigaciones", "ORGANISMO_PUBLICO"),
    (r"Carabineros", "ORGANISMO_PUBLICO"),
    (r"Tesorer[íi]a(?:\s+General)?", "ORGANISMO_PUBLICO"),
    (r"Juzgado(?:\s+de\s+Garant[íi]a|\s+Civil|\s+de\s+Letras|\s+Oral)?", "TRIBUNAL"),
    (r"Tribunal(?:\s+Oral|\s+Constitucional|\s+de\s+Juicio\s+Oral)?", "TRIBUNAL"),
    (r"Corte(?:\s+Suprema|\s+de\s+Apelaciones)?", "TRIBUNAL"),
    (r"Banco", "INSTITUCION_FINANCIERA"),
    (r"Cooperativa(?:\s+de\s+Ahorro(?:\s+y\s+Cr[ée]dito)?)?", "INSTITUCION_FINANCIERA"),
    (r"Caja\s+de\s+Compensaci[óo]n", "INSTITUCION_FINANCIERA"),
    (r"Corredora\s+de\s+Bolsa", "INSTITUCION_FINANCIERA"),
    (r"Administradora\s+General\s+de\s+Fondos", "INSTITUCION_FINANCIERA"),
    (r"Fundaci[óo]n", "ENTIDAD_SIN_FINES_DE_LUCRO"),
    (r"Corporaci[óo]n", "ENTIDAD_SIN_FINES_DE_LUCRO"),
    (r"Asociaci[óo]n(?:\s+Gremial)?", "ORGANIZACION"),
    (r"Sindicato", "ORGANIZACION"),
    (r"Confederaci[óo]n", "ORGANIZACION"),
    (r"Federaci[óo]n", "ORGANIZACION"),
    (r"Partido", "ORGANIZACION"),
    (r"Universidad", "ENTIDAD_SIN_FINES_DE_LUCRO"),
)

_RE_ENCABEZADO = re.compile(
    r"\b((?:" + "|".join(p for p, _ in _ENCABEZADOS) + r")"
    r"(?:\s+(?:de|del|de\s+la|de\s+los|para|y|e|en)\s+|\s+)?"
    r"(?:[A-ZÁÉÍÓÚÑÜ][\wÁÉÍÓÚÑÜáéíóúñü'’\-]*(?:\s+(?:de|del|la|las|los|y|e)\s+|\s+)?){0,5})",
)

_MAPA_ENCABEZADO_TIPO: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"^{patron}\b", re.IGNORECASE), tipo) for patron, tipo in _ENCABEZADOS
]


# ---------------------------------------------------------------------------
# 4. Tratamientos personales
# ---------------------------------------------------------------------------

_RE_TRATAMIENTO = re.compile(
    r"\b(?:don|doña|dona|el\s+se[ñn]or|la\s+se[ñn]ora|Sr\.|Sra\.|Srta\.)\s+"
    r"((?:[A-ZÁÉÍÓÚÑÜ][\wÁÉÍÓÚÑÜáéíóúñü'’\-]+\s*){1,4})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 5. Refinamiento por giro
# ---------------------------------------------------------------------------
#
# "Sartor Administradora General de Fondos S.A." termina en S.A., de modo que
# la regla de sufijo la clasifica como EMPRESA. Es correcto pero insuficiente:
# el giro está declarado en la propia razón social y determina un tipo más
# específico. El giro explícito prevalece sobre el sufijo genérico, porque es
# información más precisa del mismo texto, no una inferencia.

_REFINAMIENTOS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), tipo)
    for p, tipo in (
        (r"administradora\s+general\s+de\s+fondos", "INSTITUCION_FINANCIERA"),
        (r"administradora\s+de\s+fondos\s+de\s+pensiones", "INSTITUCION_FINANCIERA"),
        (r"corredora?\s+de\s+bolsa", "INSTITUCION_FINANCIERA"),
        (r"corredora?\s+de\s+seguros", "INSTITUCION_FINANCIERA"),
        (r"compañ[íi]a\s+de\s+seguros", "INSTITUCION_FINANCIERA"),
        (r"\bbanco\b", "INSTITUCION_FINANCIERA"),
        (r"cooperativa\s+de\s+ahorro", "INSTITUCION_FINANCIERA"),
        (r"caja\s+de\s+compensaci[óo]n", "INSTITUCION_FINANCIERA"),
        (r"emisora?\s+de\s+tarjetas", "INSTITUCION_FINANCIERA"),
        (r"\bfactoring\b", "INSTITUCION_FINANCIERA"),
        (r"casa\s+de\s+cambio", "INSTITUCION_FINANCIERA"),
        (r"\bexchange\b|criptomonedas?|criptoactivos?", "CRIPTOACTIVO"),
    )
)


def refinar_por_giro(texto_entidad: str, tipo_actual: str) -> tuple[str, str]:
    """Devuelve ``(tipo, senal)``. Solo refina personas jurídicas."""
    from taxonomia_uaf import TIPOS_PERSONA_JURIDICA

    if tipo_actual not in TIPOS_PERSONA_JURIDICA:
        return tipo_actual, ""
    for patron, tipo in _REFINAMIENTOS:
        if patron.search(texto_entidad):
            if tipo != tipo_actual:
                return tipo, f"giro_declarado:{patron.pattern}"
            return tipo_actual, ""
    return tipo_actual, ""


@dataclass
class CandidatoRegla:
    texto: str
    inicio: int
    fin: int
    tipo: str
    regla: str
    rut: str = ""
    senales: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------


def detectar(texto: str) -> list[CandidatoRegla]:
    """Devuelve candidatos deterministas, sin solapamiento."""
    if not texto:
        return []

    candidatos: list[CandidatoRegla] = []
    candidatos.extend(_detectar_sufijos(texto))
    candidatos.extend(_detectar_encabezados(texto))
    candidatos.extend(_detectar_tratamientos(texto))

    for cand in candidatos:
        tipo_refinado, senal = refinar_por_giro(cand.texto, cand.tipo)
        if senal:
            cand.tipo = tipo_refinado
            cand.senales.append(senal)

    return _resolver_solapamientos(candidatos)


def _detectar_sufijos(texto: str) -> list[CandidatoRegla]:
    salida: list[CandidatoRegla] = []
    for m in _RE_SUFIJO.finditer(texto):
        sufijo = m.group(2)
        tipo = "EMPRESA"
        for patron, tipo_candidato in _MAPA_SUFIJO_TIPO:
            if patron.match(sufijo.strip()):
                tipo = tipo_candidato
                break
        salida.append(
            CandidatoRegla(
                texto=m.group(0).strip(),
                inicio=m.start(),
                fin=m.start() + len(m.group(0).strip()),
                tipo=tipo,
                regla="sufijo_societario",
                senales=[f"sufijo:{sufijo.strip()}"],
            )
        )
    return salida


def _detectar_encabezados(texto: str) -> list[CandidatoRegla]:
    salida: list[CandidatoRegla] = []
    for m in _RE_ENCABEZADO.finditer(texto):
        bruto = m.group(1).strip()
        # Recorta conectores colgantes: "Ministerio de Hacienda y" -> sin la "y".
        bruto = re.sub(r"\s+(?:de|del|la|las|los|y|e|en|para)$", "", bruto).strip()
        if len(bruto) < 5:
            continue
        tipo = "ORGANIZACION"
        encabezado = ""
        for patron, tipo_candidato in _MAPA_ENCABEZADO_TIPO:
            coincide = patron.match(bruto)
            if coincide:
                tipo = tipo_candidato
                encabezado = coincide.group(0)
                break
        # "Banco" o "Corte" a secas no es una entidad nombrada.
        if plegar(bruto) == plegar(encabezado) and len(bruto.split()) < 2:
            continue
        salida.append(
            CandidatoRegla(
                texto=bruto,
                inicio=m.start(1),
                fin=m.start(1) + len(bruto),
                tipo=tipo,
                regla="encabezado_institucional",
                senales=[f"encabezado:{encabezado}"],
            )
        )
    return salida


def _detectar_tratamientos(texto: str) -> list[CandidatoRegla]:
    salida: list[CandidatoRegla] = []
    for m in _RE_TRATAMIENTO.finditer(texto):
        nombre = m.group(1).strip()
        nombre = re.sub(r"[,;:.]$", "", nombre).strip()
        if len(nombre) < 3:
            continue
        salida.append(
            CandidatoRegla(
                texto=nombre,
                inicio=m.start(1),
                fin=m.start(1) + len(nombre),
                tipo="PERSONA",
                regla="tratamiento_personal",
                senales=["tratamiento:" + m.group(0)[: m.start(1) - m.start()].strip()],
            )
        )
    return salida


def detectar_ruts(texto: str) -> list[dict[str, Any]]:
    """RUT con dígito verificador válido y su naturaleza según el rango."""
    salida: list[dict[str, Any]] = []
    for m in _RE_RUT.finditer(texto or ""):
        cuerpo = m.group(1).replace(".", "")
        dv = m.group(2).upper()
        if digito_verificador(cuerpo) != dv:
            continue
        numero = int(cuerpo)
        salida.append(
            {
                "rut": formatear_rut(cuerpo, dv),
                "inicio": m.start(),
                "fin": m.end(),
                "naturaleza_por_rango": (
                    "PERSONA_JURIDICA" if numero >= UMBRAL_RUT_JURIDICA else "PERSONA_NATURAL"
                ),
                "validado_modulo_11": True,
            }
        )
    return salida


def asociar_ruts(
    entidades: list[dict[str, Any]],
    ruts: list[dict[str, Any]],
    ventana_anterior: int = 90,
    ventana_posterior: int = 40,
) -> list[dict[str, Any]]:
    """Asocia cada RUT a una entidad. Devuelve los RUT que quedaron sin dueño.

    Muta ``entidades`` agregando el campo ``ruts``.

    Dos restricciones hacen esto seguro, y ambas surgieron de ver fallar la
    versión ingenua:

    1. **Coherencia de rango.** El tramo de numeración es determinante: bajo
       50.000.000 el RUT es de persona natural y sobre ese umbral, de persona
       jurídica. Un RUT de persona natural no se asocia jamás a una empresa,
       aunque sea la entidad más cercana. Sin esta regla, un RUT personal
       terminó pegado a "Unidad de Análisis Financiero" solo porque el nombre
       de su titular no había sido detectado.

    2. **Asimetría de ventana.** En español la convención es «Nombre, RUT X»,
       de modo que la entidad titular casi siempre precede al RUT. La ventana
       hacia atrás es más amplia que la ventana hacia adelante.

    Un RUT sin dueño coherente se devuelve sin asociar. Es preferible perder la
    asociación a atribuir un identificador tributario a la entidad equivocada.
    """
    huerfanos: list[dict[str, Any]] = []

    for rut in ruts:
        naturaleza_rut = rut["naturaleza_por_rango"]
        mejor: dict[str, Any] | None = None
        mejor_distancia = 10**9

        for ent in entidades:
            inicio = int(ent.get("inicio", -1))
            fin = int(ent.get("fin", -1))
            if inicio < 0:
                continue

            naturaleza_ent = ent.get("naturaleza", "")
            if naturaleza_ent in {"PERSONA_NATURAL", "PERSONA_JURIDICA"}:
                if naturaleza_ent != naturaleza_rut:
                    continue

            if fin <= rut["inicio"]:
                distancia = rut["inicio"] - fin
                limite = ventana_anterior
            else:
                distancia = inicio - rut["fin"]
                limite = ventana_posterior

            if 0 <= distancia <= limite and distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor = ent

        if mejor is None:
            rut["motivo_sin_asociar"] = (
                f"No hay entidad de naturaleza {naturaleza_rut} dentro de la ventana."
            )
            huerfanos.append(rut)
            continue

        mejor.setdefault("ruts", []).append(rut["rut"])
        mejor.setdefault("senales", []).append(
            f"rut_asociado:{rut['rut']}(d={mejor_distancia},rango={naturaleza_rut})"
        )

    return huerfanos


def _resolver_solapamientos(candidatos: list[CandidatoRegla]) -> list[CandidatoRegla]:
    """Ante spans solapados, gana el más largo; a igual largo, el primero."""
    ordenados = sorted(candidatos, key=lambda c: (-(c.fin - c.inicio), c.inicio))
    aceptados: list[CandidatoRegla] = []
    for cand in ordenados:
        if any(not (cand.fin <= a.inicio or cand.inicio >= a.fin) for a in aceptados):
            continue
        aceptados.append(cand)
    return sorted(aceptados, key=lambda c: c.inicio)


def es_lugar_conocido(cadena: str) -> bool:
    """Consulta el gazetteer del Monitor UAF si está disponible."""
    if not GEOGRAFIA_DISPONIBLE:
        return False
    for nombre in ("es_lugar", "es_comuna", "es_topónimo", "es_toponimo"):
        fn = getattr(_geo, nombre, None)
        if callable(fn):
            try:
                if fn(cadena):
                    return True
            except Exception:
                continue
    return False


def diagnostico() -> dict[str, Any]:
    return {
        "version_reglas": VERSION_REGLAS,
        "reconocedor_v3_disponible": RECONOCEDOR_V3_DISPONIBLE,
        "version_v3": VERSION_V3,
        "gazetteer_geografico_disponible": GEOGRAFIA_DISPONIBLE,
    }
