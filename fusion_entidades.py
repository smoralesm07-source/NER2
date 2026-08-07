#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capa L4 — Fusión con precedencia y puntaje de confianza.

Regla dura
----------
Ante conflicto de clasificación, **gana la fuente de mayor precedencia**:

    REGLA (300)  >  LLM (200)  >  GLINER (100)  >  SPACY (90)

Si la capa de reglas afirma que "Inversiones Muñoz Ltda." es EMPRESA porque
termina en un sufijo societario, esa clasificación se mantiene aunque el
adjudicador haya argumentado que se trata de una persona. Un sufijo societario
es un hecho del texto; la lectura del modelo es una inferencia. Los hechos
ganan.

El conflicto no se oculta: queda registrado en ``conflictos`` y la entidad se
marca ``requiere_validacion``, que es el campo por el que el analista entra a
revisar.

Qué produce
-----------
Un registro por entidad con la forma que ya consume ``entidades.html``
(``texto``, ``tipo``, ``naturaleza``, ``confianza_score``, ``senales``,
``ruts``, ``requiere_validacion``), más los campos nuevos del pipeline
(``rol_procesal``, ``evidencia``, ``fuentes``, ``anclaje``).
"""

from __future__ import annotations

from typing import Any

from taxonomia_uaf import (
    ROLES_ADVERSOS,
    naturaleza_de,
    normalizar_rol,
    normalizar_tipo,
    plegar,
    precedencia,
)
from validador_spans import FACTOR_ANCLAJE, ANCLAJE_EXACTO

VERSION_FUSION = "1.0.0"

#: Confianza base por fuente, antes de ajustes.
CONFIANZA_BASE: dict[str, float] = {
    "REGLA": 0.95,
    "LLM": 0.80,
    "GLINER": 0.65,
    "SPACY": 0.50,
}

CONFIANZA_LLM_DECLARADA: dict[str, float] = {"ALTA": 0.90, "MEDIA": 0.72, "BAJA": 0.50}

#: Bonificación por acuerdo entre capas independientes. Que la regla y el
#: modelo coincidan es evidencia real; que coincidan dos capas del mismo
#: modelo no lo es.
BONO_ACUERDO = 0.06

#: Penalización por desacuerdo de naturaleza jurídica.
CASTIGO_CONFLICTO = 0.25

#: Penalización cuando el rol propuesto fue inadmisible o impropio para el
#: tipo de entidad. Sin esto, una entidad cuya única señal problemática es el
#: rol puede salir con confianza máxima y bandera de revisión a la vez, que es
#: una contradicción que el analista no debería tener que interpretar.
CASTIGO_ROL_INVALIDO = 0.20

#: Penalización por artículo truncado o tras muro de pago: el contexto que
#: sustenta la clasificación puede estar cortado.
CASTIGO_ESTADO: dict[str, float] = {
    "COMPLETO": 0.00,
    "TRUNCADO": 0.10,
    "PAYWALL": 0.18,
    "VACIO": 0.40,
    "ERROR": 0.40,
}

UMBRAL_VALIDACION = 0.60


# ---------------------------------------------------------------------------
# Agrupamiento por solapamiento de spans
# ---------------------------------------------------------------------------


def _solapan(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (int(a["fin"]) <= int(b["inicio"]) or int(a["inicio"]) >= int(b["fin"]))


def agrupar_por_span(registros: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Agrupa menciones que ocupan posiciones solapadas del texto.

    Se usa solapamiento y no igualdad exacta porque las capas delimitan
    distinto: la regla captura "Banco de Chile", GLiNER puede capturar solo
    "Banco de Chile" sin el artículo, y el LLM puede incluir la coma siguiente.
    """
    ordenados = sorted(registros, key=lambda r: (int(r["inicio"]), -int(r["fin"])))
    grupos: list[list[dict[str, Any]]] = []
    for registro in ordenados:
        for grupo in grupos:
            if any(_solapan(registro, existente) for existente in grupo):
                grupo.append(registro)
                break
        else:
            grupos.append([registro])
    return grupos


# ---------------------------------------------------------------------------
# Resolución de un grupo
# ---------------------------------------------------------------------------


def resolver_grupo(grupo: list[dict[str, Any]], estado_extraccion: str) -> dict[str, Any]:
    """Consolida las menciones solapadas en una entidad."""
    # Span: el más largo del grupo. Prefiere la delimitación más completa.
    span_ganador = max(grupo, key=lambda r: int(r["fin"]) - int(r["inicio"]))

    # Tipo: por precedencia de fuente; a igual precedencia, el más frecuente.
    por_precedencia = sorted(
        grupo, key=lambda r: (-precedencia(r.get("fuente")), -_peso_interno(r))
    )
    ganador = por_precedencia[0]
    tipo, tipo_degradado = normalizar_tipo(ganador.get("tipo"))
    naturaleza = naturaleza_de(tipo)

    # Conflictos: cualquier fuente que proponga otra naturaleza jurídica.
    conflictos: list[str] = []
    naturalezas = set()
    for registro in grupo:
        tipo_i, _ = normalizar_tipo(registro.get("tipo"))
        nat_i = naturaleza_de(tipo_i)
        naturalezas.add(nat_i)
        if nat_i != naturaleza:
            conflictos.append(
                f"{registro.get('fuente', '?')} propuso {tipo_i} ({nat_i}); "
                f"prevalece {tipo} ({naturaleza}) por precedencia."
            )

    # Rol procesal: solo el adjudicador lo produce.
    rol = "SIN_ROL"
    rol_degradado = False
    evidencia = ""
    justificacion = ""
    nombre_normalizado = span_ganador.get("texto", "")
    variantes: list[str] = []

    for registro in grupo:
        if registro.get("fuente") != "LLM":
            continue
        rol_i, degradado_i = normalizar_rol(registro.get("rol_procesal"))
        rol_degradado = rol_degradado or degradado_i
        # Ante varias menciones LLM, prevalece el rol más específico.
        if _indice_rol(rol_i) > _indice_rol(rol):
            rol = rol_i
        evidencia = evidencia or str(registro.get("evidencia", ""))
        justificacion = justificacion or str(registro.get("justificacion", ""))
        nombre_normalizado = str(registro.get("nombre_normalizado") or nombre_normalizado)
        for var in registro.get("variantes", []) or []:
            if var and var not in variantes:
                variantes.append(var)

    # Un lugar, un monto o una fecha no pueden tener calidad procesal: no son
    # sujetos de derecho. Si el modelo asignó un rol a una de estas, el rol se
    # anula, pero el hecho de que lo haya intentado queda registrado porque es
    # síntoma de que confundió el tipo.
    rol_impropio = False
    if naturaleza not in {"PERSONA_NATURAL", "PERSONA_JURIDICA"} and rol != "SIN_ROL":
        conflictos.append(
            f"Se propuso el rol {rol} para una entidad de tipo {tipo}, que no es "
            "sujeto de derecho. El rol fue anulado."
        )
        rol = "SIN_ROL"
        rol_impropio = True

    # Confianza. Un rol degradado o impropio también la castiga: si el modelo
    # erró al punto de proponer un rol inadmisible para esta entidad, su
    # juicio sobre ella completa merece menos crédito, no el máximo.
    puntaje = _calcular_confianza(
        grupo, bool(conflictos), estado_extraccion, rol_degradado or rol_impropio
    )

    fuentes = sorted({str(r.get("fuente", "?")) for r in grupo})
    senales: list[str] = []
    for registro in grupo:
        for senal in registro.get("senales", []) or []:
            if senal not in senales:
                senales.append(senal)

    ruts: list[str] = []
    for registro in grupo:
        for rut in registro.get("ruts", []) or []:
            if rut not in ruts:
                ruts.append(rut)

    anclajes = {str(r.get("anclaje", ANCLAJE_EXACTO)) for r in grupo}
    anclaje = min(anclajes, key=lambda a: FACTOR_ANCLAJE.get(a, 0.5))

    requiere_validacion = bool(
        conflictos
        or rol_degradado
        or rol_impropio
        or tipo_degradado
        or puntaje < UMBRAL_VALIDACION
        or (rol in ROLES_ADVERSOS and puntaje < 0.85)
    )

    motivos: list[str] = []
    if conflictos:
        motivos.append("conflicto de naturaleza entre capas")
    if rol_degradado:
        motivos.append("el rol procesal devuelto no era admisible y se degradó")
    if rol_impropio:
        motivos.append("se asignó rol procesal a una entidad que no es sujeto de derecho")
    if tipo_degradado:
        motivos.append("tipo no reconocido en la taxonomía")
    if puntaje < UMBRAL_VALIDACION:
        motivos.append(f"confianza {puntaje:.2f} bajo el umbral {UMBRAL_VALIDACION}")
    if rol in ROLES_ADVERSOS and puntaje < 0.85:
        motivos.append("rol adverso sin confianza suficiente para publicar sin revisión")

    return {
        "texto": span_ganador.get("texto", ""),
        "nombre_normalizado": nombre_normalizado,
        "inicio": int(span_ganador["inicio"]),
        "fin": int(span_ganador["fin"]),
        "tipo": tipo,
        "naturaleza": naturaleza,
        "rol_procesal": rol,
        "confianza_score": round(puntaje, 3),
        "fuentes": fuentes,
        "anclaje": anclaje,
        "evidencia": evidencia,
        "justificacion": justificacion,
        "variantes": variantes,
        "ruts": ruts,
        "senales": senales,
        "conflictos": conflictos,
        "requiere_validacion": requiere_validacion,
        "motivos_validacion": motivos,
        "n_menciones": len(grupo),
    }


def _peso_interno(registro: dict[str, Any]) -> float:
    if registro.get("fuente") == "LLM":
        return CONFIANZA_LLM_DECLARADA.get(str(registro.get("confianza", "MEDIA")).upper(), 0.7)
    return float(registro.get("score", 0.5) or 0.5)


def _indice_rol(rol: str) -> int:
    from taxonomia_uaf import ROLES_PROCESALES

    try:
        return ROLES_PROCESALES.index(rol)
    except ValueError:
        return 0


def _calcular_confianza(
    grupo: list[dict[str, Any]],
    hay_conflicto: bool,
    estado_extraccion: str,
    rol_invalido: bool = False,
) -> float:
    base = 0.0
    for registro in grupo:
        fuente = str(registro.get("fuente", "")).upper()
        if fuente == "LLM":
            valor = CONFIANZA_LLM_DECLARADA.get(
                str(registro.get("confianza", "MEDIA")).upper(), 0.72
            )
        elif fuente == "GLINER":
            # El score de GLiNER se mezcla con la base para no premiar en
            # exceso una predicción apenas sobre el umbral.
            valor = (CONFIANZA_BASE["GLINER"] + float(registro.get("score", 0.5))) / 2
        else:
            valor = CONFIANZA_BASE.get(fuente, 0.5)
        base = max(base, valor)

    fuentes = {str(r.get("fuente", "")).upper() for r in grupo}
    independientes = fuentes & {"REGLA", "LLM", "GLINER", "SPACY"}
    if len(independientes) >= 2 and not hay_conflicto:
        base += BONO_ACUERDO * (len(independientes) - 1)

    if hay_conflicto:
        base -= CASTIGO_CONFLICTO

    if rol_invalido:
        base -= CASTIGO_ROL_INVALIDO

    anclaje_peor = min(
        (FACTOR_ANCLAJE.get(str(r.get("anclaje", ANCLAJE_EXACTO)), 0.5) for r in grupo),
        default=1.0,
    )
    base *= anclaje_peor

    base -= CASTIGO_ESTADO.get(str(estado_extraccion).upper(), 0.0)

    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# Correferencia entre entidades ya fusionadas
# ---------------------------------------------------------------------------


def unificar_correferentes(entidades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fusiona entidades que designan al mismo referente.

    Solo se unifica cuando el adjudicador declaró la equivalencia (vía
    ``variantes`` o ``nombre_normalizado``) **y** ambas comparten naturaleza.
    No se unifica por prefijo compartido: esa heurística es la que colapsaría
    "Sartor Finance Group" con "Sartor AGF S.A.", que son personas jurídicas
    distintas.
    """
    grupos: list[list[dict[str, Any]]] = []

    for entidad in entidades:
        claves = {plegar(entidad.get("nombre_normalizado") or entidad["texto"])}
        claves |= {plegar(v) for v in entidad.get("variantes", []) if v}
        claves.add(plegar(entidad["texto"]))

        for grupo in grupos:
            if entidad["naturaleza"] != grupo[0]["naturaleza"]:
                continue
            claves_grupo: set[str] = set()
            for miembro in grupo:
                claves_grupo.add(plegar(miembro.get("nombre_normalizado") or miembro["texto"]))
                claves_grupo.add(plegar(miembro["texto"]))
                claves_grupo |= {plegar(v) for v in miembro.get("variantes", []) if v}
            if claves & claves_grupo:
                grupo.append(entidad)
                break
        else:
            grupos.append([entidad])

    salida: list[dict[str, Any]] = []
    for grupo in grupos:
        if len(grupo) == 1:
            salida.append(grupo[0])
            continue
        principal = max(grupo, key=lambda e: (e["confianza_score"], len(e["texto"])))
        principal = dict(principal)
        principal["menciones"] = [
            {"texto": m["texto"], "inicio": m["inicio"], "fin": m["fin"]} for m in grupo
        ]
        principal["n_menciones"] = sum(m.get("n_menciones", 1) for m in grupo)
        for miembro in grupo:
            if miembro is principal:
                continue
            for campo in ("ruts", "senales", "variantes", "conflictos"):
                for valor in miembro.get(campo, []) or []:
                    if valor not in principal.setdefault(campo, []):
                        principal[campo].append(valor)
            if _indice_rol(miembro.get("rol_procesal", "SIN_ROL")) > _indice_rol(
                principal.get("rol_procesal", "SIN_ROL")
            ):
                principal["rol_procesal"] = miembro["rol_procesal"]
                principal["evidencia"] = miembro.get("evidencia") or principal.get("evidencia", "")
        principal["requiere_validacion"] = any(m["requiere_validacion"] for m in grupo)
        salida.append(principal)

    return salida


# ---------------------------------------------------------------------------
# Entrada pública
# ---------------------------------------------------------------------------


def fusionar(
    registros: list[dict[str, Any]], estado_extraccion: str = "COMPLETO"
) -> list[dict[str, Any]]:
    """Agrupa, resuelve y unifica. Devuelve entidades ordenadas por aparición."""
    validos = [
        r
        for r in registros
        if isinstance(r.get("inicio"), int)
        and isinstance(r.get("fin"), int)
        and int(r["fin"]) > int(r["inicio"])
    ]
    if not validos:
        return []

    resueltas = [resolver_grupo(g, estado_extraccion) for g in agrupar_por_span(validos)]
    unificadas = unificar_correferentes(resueltas)
    return sorted(unificadas, key=lambda e: e["inicio"])


def resumir(entidades: list[dict[str, Any]]) -> dict[str, Any]:
    """Conteos para el encabezado del informe."""
    por_naturaleza: dict[str, int] = {}
    por_tipo: dict[str, int] = {}
    for entidad in entidades:
        por_naturaleza[entidad["naturaleza"]] = por_naturaleza.get(entidad["naturaleza"], 0) + 1
        por_tipo[entidad["tipo"]] = por_tipo.get(entidad["tipo"], 0) + 1

    return {
        "total": len(entidades),
        "conteo_por_naturaleza": por_naturaleza,
        "conteo_por_tipo": por_tipo,
        "requieren_validacion": sum(1 for e in entidades if e["requiere_validacion"]),
        "con_rol_adverso": sum(1 for e in entidades if e["rol_procesal"] in ROLES_ADVERSOS),
        "con_rut": sum(1 for e in entidades if e.get("ruts")),
    }
