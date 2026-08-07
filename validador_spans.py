#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verificación de anclaje textual — la barrera anti-alucinación.

La única objeción seria contra usar un LLM para NER es que puede inventar
entidades que no están en el texto. Este módulo la neutraliza con una regla
mecánica: **toda entidad debe venir acompañada de su cadena literal y su
offset, y se verifica en Python que la cadena esté efectivamente ahí.** Lo que
no se verifica, no se emite.

El modelo no puede eludir esta comprobación argumentando: es una igualdad de
strings, no un juicio.

Tolerancias
-----------
Se aceptan tres niveles de anclaje, en orden decreciente de confianza:

``EXACTO``
    ``texto[inicio:inicio+len(span)] == span``. Sin más.

``REUBICADO``
    La cadena existe en el texto pero en otra posición. Los modelos son
    notoriamente imprecisos contando caracteres, así que este caso es
    frecuente y no indica alucinación. Se reubica al match más cercano al
    offset declarado y se conserva.

``NORMALIZADO``
    La cadena solo coincide tras plegar espacios en blanco y tildes. Ocurre
    cuando el modelo re-acentúa un nombre o colapsa un salto de línea. Se
    conserva usando la superficie real del texto, no la que devolvió el
    modelo, y se marca para revisión.

Cualquier otro caso es ``NO_ANCLADO`` y se rechaza.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

VERSION_VALIDADOR = "1.0.0"

ANCLAJE_EXACTO = "EXACTO"
ANCLAJE_REUBICADO = "REUBICADO"
ANCLAJE_NORMALIZADO = "NORMALIZADO"
ANCLAJE_FALLIDO = "NO_ANCLADO"

#: Penalización aplicada a la confianza según la calidad del anclaje.
FACTOR_ANCLAJE: dict[str, float] = {
    ANCLAJE_EXACTO: 1.00,
    ANCLAJE_REUBICADO: 0.98,
    ANCLAJE_NORMALIZADO: 0.85,
}


@dataclass
class ResultadoAnclaje:
    anclado: bool
    metodo: str
    inicio: int
    fin: int
    superficie: str
    detalle: str = ""


def _plegar_conservando_longitud(texto: str) -> str:
    """Quita diacríticos sin alterar el número de caracteres.

    Es indispensable para que los offsets calculados sobre la versión plegada
    sigan siendo válidos sobre el texto original. ``NFD`` descompone y agrega
    caracteres, por eso se recompone a NFC tras filtrar las marcas, carácter a
    carácter.
    """
    salida: list[str] = []
    for caracter in texto:
        descompuesto = unicodedata.normalize("NFD", caracter)
        base = "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")
        salida.append(base[:1] if base else caracter)
    return "".join(salida).casefold()


def anclar(texto: str, span: str, inicio_declarado: int | None = None) -> ResultadoAnclaje:
    """Localiza ``span`` dentro de ``texto``."""
    if not texto or not span:
        return ResultadoAnclaje(False, ANCLAJE_FALLIDO, -1, -1, "", "Span o texto vacío.")

    span = span.strip()
    largo = len(span)

    # Nivel 1: coincidencia exacta en la posición declarada.
    if inicio_declarado is not None and 0 <= inicio_declarado <= len(texto) - largo:
        if texto[inicio_declarado : inicio_declarado + largo] == span:
            return ResultadoAnclaje(
                True, ANCLAJE_EXACTO, inicio_declarado, inicio_declarado + largo, span
            )

    # Nivel 2: la cadena existe en otra posición.
    posiciones = [m.start() for m in re.finditer(re.escape(span), texto)]
    if posiciones:
        referencia = inicio_declarado if inicio_declarado is not None else 0
        elegida = min(posiciones, key=lambda p: abs(p - referencia))
        return ResultadoAnclaje(
            True,
            ANCLAJE_EXACTO if len(posiciones) == 1 and inicio_declarado is None
            else ANCLAJE_REUBICADO,
            elegida,
            elegida + largo,
            span,
            f"{len(posiciones)} ocurrencia(s); offset declarado {inicio_declarado}.",
        )

    # Nivel 3: coincidencia tras plegar tildes y espacios.
    texto_plegado = _plegar_conservando_longitud(texto)
    span_plegado = _plegar_conservando_longitud(span)
    posiciones = [m.start() for m in re.finditer(re.escape(span_plegado), texto_plegado)]
    if posiciones:
        referencia = inicio_declarado if inicio_declarado is not None else 0
        elegida = min(posiciones, key=lambda p: abs(p - referencia))
        superficie_real = texto[elegida : elegida + largo]
        return ResultadoAnclaje(
            True,
            ANCLAJE_NORMALIZADO,
            elegida,
            elegida + largo,
            superficie_real,
            f"El modelo devolvió «{span}»; en el texto dice «{superficie_real}».",
        )

    # Nivel 4 (whitespace): el modelo colapsó un salto de línea o doble espacio.
    patron_flexible = r"\s+".join(re.escape(p) for p in span.split())
    coincidencia = re.search(patron_flexible, texto)
    if coincidencia:
        return ResultadoAnclaje(
            True,
            ANCLAJE_NORMALIZADO,
            coincidencia.start(),
            coincidencia.end(),
            coincidencia.group(0),
            "Coincidencia tras normalizar espacios en blanco.",
        )

    return ResultadoAnclaje(
        False,
        ANCLAJE_FALLIDO,
        -1,
        -1,
        "",
        f"La cadena «{span}» no aparece en el texto del artículo.",
    )


def validar_lote(
    texto: str, entidades: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ancla una lista de entidades. Devuelve ``(aceptadas, rechazadas)``.

    Muta cada entidad aceptada agregando ``inicio``, ``fin``, ``texto`` (la
    superficie real) y ``anclaje``.
    """
    aceptadas: list[dict[str, Any]] = []
    rechazadas: list[dict[str, Any]] = []

    for entidad in entidades:
        span = str(entidad.get("span_exacto") or entidad.get("texto") or "")
        declarado = entidad.get("offset_inicio", entidad.get("inicio"))
        try:
            declarado = int(declarado) if declarado is not None else None
        except (TypeError, ValueError):
            declarado = None

        resultado = anclar(texto, span, declarado)

        if not resultado.anclado:
            entidad["motivo_rechazo"] = resultado.detalle
            entidad["anclaje"] = ANCLAJE_FALLIDO
            rechazadas.append(entidad)
            continue

        entidad["texto"] = resultado.superficie
        entidad["inicio"] = resultado.inicio
        entidad["fin"] = resultado.fin
        entidad["anclaje"] = resultado.metodo
        if resultado.detalle and resultado.metodo != ANCLAJE_EXACTO:
            entidad.setdefault("senales", []).append(f"anclaje:{resultado.detalle}")
        aceptadas.append(entidad)

    return aceptadas, rechazadas


def validar_evidencia(texto: str, evidencia: str) -> bool:
    """La frase de evidencia también debe existir en el artículo.

    Sin esta comprobación el modelo puede anclar correctamente el nombre y aun
    así inventar la oración que supuestamente justifica el rol procesal, que es
    el dato con consecuencias.
    """
    if not evidencia:
        return False
    return anclar(texto, evidencia.strip()).anclado
