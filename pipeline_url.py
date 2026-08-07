#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orquestador — URL de prensa → entidades clasificadas.

Cascada
-------
    L0  extractor_articulo   descarga, limpia boilerplate, diagnostica el estado
    L1  capa_reglas          sufijos societarios, encabezados, RUT módulo 11
    L2  capa_gliner          candidatos zero-shot con la taxonomía UAF
    L3  adjudicador_llm      naturaleza, correferencia, rol procesal, relaciones
    L3b validador_spans      descarta lo que no exista literalmente en el texto
    L4  fusion_entidades     precedencia REGLA > LLM > GLINER, confianza, cola

Cada capa degrada por separado. Sin GLiNER instalado el pipeline funciona con
reglas y adjudicador; sin credencial de API funciona con reglas y GLiNER. La
salida siempre declara qué capas corrieron, en ``capas``.

Uso
---
    python pipeline_url.py --url https://... --salida resultado.json
    python pipeline_url.py --archivo nota.html --sin-llm
    python pipeline_url.py --texto "..." --simular
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import capa_gliner
import capa_gliner2
import capa_reglas
import fusion_entidades
import validador_spans
from adjudicador_llm import ErrorAdjudicador, adjudicar
from extractor_articulo import ArticuloExtraido, extraer_articulo, extraer_desde_html
from taxonomia_uaf import VERSION_TAXONOMIA

VERSION_PIPELINE = "1.1.0"

#: Motor de la capa L2. "v2" usa gliner2 (fastino-ai) con esquema descriptivo;
#: "v1" usa el GLiNER original. Ambos exponen la misma interfaz y devuelven
#: CandidatoGliner, de modo que L3 y L4 no distinguen cuál corrió.
BACKEND_L2 = os.environ.get("GLINER_BACKEND", "v2").lower()


def _motor_l2():
    """Devuelve el módulo de la capa L2 según la configuración, con respaldo.

    Si el motor elegido no está instalado, se prueba el otro antes de caer a
    spaCy. Un despliegue con solo uno de los dos paquetes debe funcionar sin
    tocar configuración.
    """
    preferido = capa_gliner2 if BACKEND_L2 == "v2" else capa_gliner
    alterno = capa_gliner if BACKEND_L2 == "v2" else capa_gliner2
    if preferido.disponible():
        return preferido
    if alterno.disponible():
        return alterno
    return preferido  # para que su mensaje de error llegue a las advertencias


def analizar_articulo(
    articulo: ArticuloExtraido,
    usar_gliner: bool = True,
    usar_llm: bool = True,
    simular_llm: bool = False,
    umbral_gliner: float = capa_gliner.UMBRAL_POR_DEFECTO,
) -> dict[str, Any]:
    """Ejecuta L1–L4 sobre un artículo ya extraído."""
    inicio_reloj = time.time()
    texto = articulo.texto_analizable
    capas: dict[str, Any] = {}
    advertencias: list[str] = list(articulo.advertencias)

    if not articulo.utilizable:
        return _salida(articulo, [], [], {}, capas, advertencias, inicio_reloj)

    # --- L1: reglas deterministas -----------------------------------------
    candidatos: list[dict[str, Any]] = []
    reglas = capa_reglas.detectar(texto)
    for cand in reglas:
        candidatos.append(
            {
                "texto": cand.texto,
                "inicio": cand.inicio,
                "fin": cand.fin,
                "tipo": cand.tipo,
                "fuente": "REGLA",
                "senales": list(cand.senales) + [f"regla:{cand.regla}"],
                "anclaje": validador_spans.ANCLAJE_EXACTO,
            }
        )
    ruts = capa_reglas.detectar_ruts(texto)
    capas["L1_reglas"] = {
        "ejecutada": True,
        "candidatos": len(reglas),
        "ruts_validos": len(ruts),
        **capa_reglas.diagnostico(),
    }

    # --- L2: GLiNER --------------------------------------------------------
    if usar_gliner:
        motor = _motor_l2()
        gliner_cands = motor.detectar(texto, umbral=umbral_gliner)
        fuente = "GLINER"
        if not gliner_cands and not motor.disponible():
            gliner_cands = capa_gliner.detectar_con_respaldo_spacy(texto)
            fuente = "SPACY"
            if gliner_cands:
                advertencias.append(
                    "GLiNER no disponible; se usó spaCy como respaldo con "
                    "tipos gruesos."
                )
            elif motor.error_carga():
                advertencias.append(motor.error_carga())

        for cand in gliner_cands:
            candidatos.append(
                {
                    "texto": cand.texto,
                    "inicio": cand.inicio,
                    "fin": cand.fin,
                    "tipo": cand.tipo,
                    "fuente": fuente,
                    "score": cand.score,
                    "senales": [f"etiqueta:{cand.etiqueta_original}"],
                    "anclaje": validador_spans.ANCLAJE_EXACTO,
                }
            )
        capas["L2_gliner"] = {
            "ejecutada": True,
            "fuente_efectiva": fuente,
            "candidatos": len(gliner_cands),
            **motor.diagnostico(),
        }
    else:
        capas["L2_gliner"] = {"ejecutada": False, "motivo": "desactivada por parámetro"}

    # --- L3: adjudicación LLM ---------------------------------------------
    resultado_llm: dict[str, Any] = {}
    rechazadas: list[dict[str, Any]] = []

    if usar_llm:
        try:
            resultado_llm = adjudicar(
                texto,
                candidatos=candidatos,
                metadatos={
                    "medio": articulo.medio,
                    "titulo": articulo.titulo,
                    "fecha_publicacion": articulo.fecha_publicacion,
                    "estado_extraccion": articulo.estado_extraccion,
                },
                simular=simular_llm,
            )
        except ErrorAdjudicador as exc:
            advertencias.append(f"Adjudicación no ejecutada: {exc}")
            capas["L3_adjudicador"] = {"ejecutada": False, "error": str(exc)}
        else:
            crudas = list(resultado_llm.get("entidades", []))
            aceptadas, rechazadas = validador_spans.validar_lote(texto, crudas)

            for ent in aceptadas:
                ent["fuente"] = "LLM"
                candidatos.append(ent)

            # La evidencia también se verifica: anclar bien el nombre e
            # inventar la frase que sustenta el rol es el error con
            # consecuencias.
            sin_evidencia = 0
            for ent in aceptadas:
                if not validador_spans.validar_evidencia(texto, str(ent.get("evidencia", ""))):
                    sin_evidencia += 1
                    ent["evidencia"] = ""
                    ent.setdefault("senales", []).append(
                        "evidencia_no_verificada_en_el_texto"
                    )

            capas["L3_adjudicador"] = {
                "ejecutada": True,
                "simulado": bool(resultado_llm.get("_simulado")),
                "modelo": resultado_llm.get("_modelo", ""),
                "propuestas": len(crudas),
                "ancladas": len(aceptadas),
                "rechazadas_por_anclaje": len(rechazadas),
                "evidencias_descartadas": sin_evidencia,
                "uso_tokens": resultado_llm.get("_uso", {}),
            }
            if rechazadas:
                advertencias.append(
                    f"{len(rechazadas)} entidad(es) propuestas por el modelo no "
                    "existen literalmente en el artículo y fueron descartadas."
                )
    else:
        capas["L3_adjudicador"] = {"ejecutada": False, "motivo": "desactivada por parámetro"}

    # --- L4: fusión --------------------------------------------------------
    entidades = fusion_entidades.fusionar(candidatos, articulo.estado_extraccion)
    ruts_huerfanos = capa_reglas.asociar_ruts(entidades, ruts)
    capas["L4_fusion"] = {
        "ejecutada": True,
        "entidades": len(entidades),
        "ruts_sin_asociar": len(ruts_huerfanos),
    }
    if ruts_huerfanos:
        advertencias.append(
            f"{len(ruts_huerfanos)} RUT válido(s) no se pudieron asociar a ninguna "
            "entidad coherente: "
            + ", ".join(r["rut"] for r in ruts_huerfanos)
        )

    relaciones = _anclar_relaciones(texto, resultado_llm.get("relaciones", []))

    return _salida(
        articulo,
        entidades,
        relaciones,
        resultado_llm,
        capas,
        advertencias,
        inicio_reloj,
        rechazadas,
    )


def _anclar_relaciones(texto: str, relaciones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conserva solo relaciones cuya frase de sustento exista en el artículo."""
    salida: list[dict[str, Any]] = []
    for rel in relaciones:
        evidencia = str(rel.get("evidencia", ""))
        if validador_spans.validar_evidencia(texto, evidencia):
            rel["evidencia_verificada"] = True
            salida.append(rel)
    return salida


def _salida(
    articulo: ArticuloExtraido,
    entidades: list[dict[str, Any]],
    relaciones: list[dict[str, Any]],
    resultado_llm: dict[str, Any],
    capas: dict[str, Any],
    advertencias: list[str],
    inicio_reloj: float,
    rechazadas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version_pipeline": VERSION_PIPELINE,
        "version_taxonomia": VERSION_TAXONOMIA,
        "articulo": articulo.a_dict(),
        "texto_analizado": articulo.texto_analizable,
        "entidades": entidades,
        "relaciones": relaciones,
        "delitos_mencionados": resultado_llm.get("delitos_mencionados", []),
        "sintesis": resultado_llm.get("sintesis", ""),
        "hay_procedimiento": resultado_llm.get("hay_procedimiento", False),
        "resumen": fusion_entidades.resumir(entidades),
        "capas": capas,
        "rechazadas_por_anclaje": rechazadas or [],
        "advertencias": advertencias,
        "segundos": round(time.time() - inicio_reloj, 2),
    }


def analizar_url(url: str, **opciones: Any) -> dict[str, Any]:
    return analizar_articulo(extraer_articulo(url), **opciones)


def analizar_html(html: str, url: str = "", **opciones: Any) -> dict[str, Any]:
    return analizar_articulo(extraer_desde_html(html, url=url), **opciones)


def analizar_texto(texto: str, titulo: str = "", **opciones: Any) -> dict[str, Any]:
    articulo = ArticuloExtraido(
        url="",
        texto=texto,
        titulo=titulo,
        estado_extraccion="COMPLETO",
        n_caracteres=len(texto),
        extractor="texto-directo",
    )
    return analizar_articulo(articulo, **opciones)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analiza una noticia y extrae entidades clasificadas."
    )
    origen = parser.add_mutually_exclusive_group(required=True)
    origen.add_argument("--url", help="URL de la noticia")
    origen.add_argument("--archivo", help="Ruta a un HTML local")
    origen.add_argument("--texto", help="Texto plano del artículo")

    parser.add_argument("--salida", help="Ruta del JSON de salida")
    parser.add_argument("--sin-gliner", action="store_true", help="Omitir la capa L2")
    parser.add_argument("--sin-llm", action="store_true", help="Omitir la capa L3")
    parser.add_argument(
        "--simular",
        action="store_true",
        help="Ejecutar L3 en modo simulado, sin llamar a la API",
    )
    parser.add_argument(
        "--umbral-gliner",
        type=float,
        default=capa_gliner.UMBRAL_POR_DEFECTO,
        help=f"Umbral de GLiNER (por defecto {capa_gliner.UMBRAL_POR_DEFECTO})",
    )
    parser.add_argument(
        "--resumen", action="store_true", help="Imprimir un resumen legible en vez del JSON"
    )

    args = parser.parse_args(argv)

    opciones = {
        "usar_gliner": not args.sin_gliner,
        "usar_llm": not args.sin_llm,
        "simular_llm": args.simular,
        "umbral_gliner": args.umbral_gliner,
    }

    if args.url:
        resultado = analizar_url(args.url, **opciones)
    elif args.archivo:
        with open(args.archivo, encoding="utf-8", errors="replace") as fh:
            resultado = analizar_html(fh.read(), url=args.archivo, **opciones)
    else:
        resultado = analizar_texto(args.texto, **opciones)

    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as fh:
            json.dump(resultado, fh, ensure_ascii=False, indent=2)
        print(f"Escrito: {args.salida}", file=sys.stderr)

    if args.resumen:
        _imprimir_resumen(resultado)
    elif not args.salida:
        print(json.dumps(resultado, ensure_ascii=False, indent=2))

    return 0 if resultado["articulo"]["utilizable"] else 1


def _imprimir_resumen(resultado: dict[str, Any]) -> None:
    art = resultado["articulo"]
    res = resultado["resumen"]
    print(f"\n{art.get('titulo') or '(sin título)'}")
    print(f"{art.get('medio', '')}  ·  {art.get('fecha_publicacion', '')}")
    print(f"Extracción: {art['estado_extraccion']} ({art['n_caracteres']} caracteres)")
    print(f"Entidades: {res['total']}  ·  requieren validación: {res['requieren_validacion']}")
    print(f"Por naturaleza: {res['conteo_por_naturaleza']}")
    print("-" * 72)
    for ent in resultado["entidades"]:
        marca = "⚠ " if ent["requiere_validacion"] else "  "
        rol = "" if ent["rol_procesal"] == "SIN_ROL" else f"  [{ent['rol_procesal']}]"
        ruts = f"  RUT {', '.join(ent['ruts'])}" if ent.get("ruts") else ""
        print(
            f"{marca}{ent['texto'][:44]:<46} {ent['tipo']:<28} "
            f"{ent['confianza_score']:.2f}  {'+'.join(ent['fuentes'])}{rol}{ruts}"
        )
    for adv in resultado["advertencias"]:
        print(f"\n! {adv}")


if __name__ == "__main__":
    raise SystemExit(main())
