#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera ``demo_resultado.json`` sin consumir la API.

Sirve para dos cosas:

1. Revisar la interfaz (``analizar_url.html`` → «Abrir JSON») antes de tener
   credencial configurada.
2. Ver, sobre un caso concreto, qué hace cada barrera del pipeline. La
   adjudicación simulada incluye a propósito tres defectos que un modelo real
   puede cometer, y los tres se detienen antes de la salida.

Ejecución:  python generar_demo.py
"""

from __future__ import annotations

import json
from unittest import mock

import pipeline_url
from extractor_articulo import extraer_desde_html

RUTA_FIXTURE = "fixtures/nota_ejemplo.html"
RUTA_SALIDA = "demo_resultado.json"


def _adjudicacion_simulada(texto, candidatos=None, metadatos=None, **kwargs):
    """Salida plausible de Claude, con tres defectos deliberados."""

    def ent(span, tipo, rol, evidencia, justificacion, confianza="ALTA", variantes=None):
        return {
            "span_exacto": span,
            "offset_inicio": texto.find(span),
            "nombre_normalizado": span,
            "tipo": tipo,
            "rol_procesal": rol,
            "evidencia": evidencia,
            "justificacion": justificacion,
            "confianza": confianza,
            "variantes": variantes or [],
        }

    return {
        "entidades": [
            ent(
                "Ricardo Andrés Bustamante Leiva", "PERSONA", "FORMALIZADO",
                "formalizó a Ricardo Andrés Bustamante Leiva",
                "Nombre y apellidos de persona natural, con calidad procesal explícita.",
                variantes=["Ricardo Bustamante"],
            ),
            ent(
                "Marcela Ortiz Vega", "PERSONA", "FORMALIZADO",
                "y a la empresaria Marcela Ortiz Vega por el delito de lavado de activos",
                "Persona natural individualizada con nombre y dos apellidos.",
                variantes=["Marcela Ortiz"],
            ),
            ent(
                "Julio Sepúlveda", "PERSONA", "DEFENSA",
                "El abogado defensor, don Julio Sepúlveda, sostuvo",
                "Persona natural en calidad de defensor, no de investigado.",
            ),
            ent(
                "Puente Alto", "LUGAR", "SIN_ROL",
                "Juzgado de Garantía de Puente Alto",
                "Comuna de la Región Metropolitana.",
            ),
            ent(
                "San Ramón", "LUGAR", "SIN_ROL",
                "ambas con domicilio en la comuna de San Ramón",
                "Comuna, no persona, pese a tener forma de nombre propio.",
            ),
            ent(
                # El caso trampa: nombre de comuna idéntico al de un expresidente.
                "Pedro Aguirre Cerda", "LUGAR", "SIN_ROL",
                "El caso quedó radicado en la comuna de Pedro Aguirre Cerda",
                "El contexto territorial («la comuna de») resuelve la ambigüedad "
                "con el antropónimo.",
            ),
            ent(
                "Sartor Finance Group", "EMPRESA", "MENCIONADO",
                "entidad distinta de Sartor Finance Group",
                "Persona jurídica distinta de Sartor AGF S.A.; el propio texto lo "
                "precisa.",
            ),
            # DEFECTO 1 — contradice a la capa de reglas.
            ent(
                "Inversiones Bustamante y Compañía Limitada", "PERSONA", "INVESTIGADO",
                "las operaciones se habrían canalizado a través de Inversiones Bustamante",
                "Contiene el apellido del formalizado.",
            ),
            # DEFECTO 2 — entidad que no está en el artículo.
            ent(
                "Banco Santander Chile", "INSTITUCION_FINANCIERA", "REPORTANTE",
                "el banco remitió reportes de operaciones sospechosas",
                "Entidad reportante mencionada en el artículo.",
                confianza="MEDIA",
            ),
            # DEFECTO 3 — rol que imputa responsabilidad penal.
            ent(
                "Comercializadora del Maipo SpA", "EMPRESA", "LAVADOR",
                "y de Comercializadora del Maipo SpA",
                "Sociedad utilizada para canalizar los fondos.",
            ),
        ],
        "relaciones": [
            {
                "origen": "Fiscalía Regional Metropolitana Sur",
                "tipo_relacion": "FORMALIZA_A",
                "destino": "Ricardo Andrés Bustamante Leiva",
                "evidencia": "acogió este martes la solicitud de la Fiscalía Regional "
                             "Metropolitana Sur y formalizó a Ricardo Andrés Bustamante Leiva",
            },
            {
                "origen": "Unidad de Análisis Financiero",
                "tipo_relacion": "REPORTA_A",
                "destino": "Ministerio Público",
                "evidencia": "La Unidad de Análisis Financiero remitió antecedentes al "
                             "Ministerio Público",
            },
            {
                "origen": "Julio Sepúlveda",
                "tipo_relacion": "DEFIENDE_A",
                "destino": "Marcela Ortiz Vega",
                "evidencia": "El abogado defensor, don Julio Sepúlveda, sostuvo que su "
                             "representada, Marcela Ortiz",
            },
            {
                # Relación con evidencia inventada: debe descartarse.
                "origen": "Ricardo Andrés Bustamante Leiva",
                "tipo_relacion": "BENEFICIARIO_FINAL_DE",
                "destino": "Comercializadora del Maipo SpA",
                "evidencia": "figura como beneficiario final según el registro societario",
            },
        ],
        "delitos_mencionados": ["lavado de activos"],
        "sintesis": (
            "El Juzgado de Garantía de Puente Alto formalizó a dos personas por lavado "
            "de activos tras una investigación originada en reportes de operaciones "
            "sospechosas remitidos por la UAF al Ministerio Público."
        ),
        "hay_procedimiento": True,
        "_simulado": False,
        "_uso": {"input_tokens": 2380, "output_tokens": 1140, "cache_read_input_tokens": 1810},
        "_modelo": "simulado-para-demostracion",
    }


def main() -> int:
    with open(RUTA_FIXTURE, encoding="utf-8") as fh:
        html = fh.read()

    articulo = extraer_desde_html(html, url="https://diario-ejemplo.cl/nacional/formalizacion")

    with mock.patch.object(pipeline_url, "adjudicar", _adjudicacion_simulada):
        resultado = pipeline_url.analizar_articulo(articulo, usar_gliner=False)

    with open(RUTA_SALIDA, "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, ensure_ascii=False, indent=2)

    res = resultado["resumen"]
    l3 = resultado["capas"]["L3_adjudicador"]
    print(f"Escrito: {RUTA_SALIDA}")
    print(f"  entidades          {res['total']}")
    print(f"  por naturaleza     {res['conteo_por_naturaleza']}")
    print(f"  por revisar        {res['requieren_validacion']}")
    print(f"  propuestas L3      {l3['propuestas']}")
    print(f"  rechazadas anclaje {l3['rechazadas_por_anclaje']}")
    print(f"  relaciones         {len(resultado['relaciones'])} de 4 propuestas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
