#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Suite de regresión del pipeline.

Qué se prueba y qué no
----------------------
Estas pruebas verifican **la mecánica del pipeline**: que el anclaje descarte
lo que no está en el texto, que la precedencia se respete, que los roles
prohibidos se degraden, que el chunking remapee offsets correctamente y que
cada capa degrade sin arrastrar a las demás.

Estas pruebas NO miden desempeño de reconocimiento. Pasar los 40 casos no dice
nada sobre precision/recall en prensa real; para eso hace falta el gold
standard anotado descrito en README_PIPELINE.md. Confundir una cosa con la otra
es el error que produce un F1 de 1.000 que no significa nada.

Ejecución:  python -m unittest test_pipeline_url -v
"""

from __future__ import annotations

import unittest
from unittest import mock

import capa_gliner
import capa_gliner2
import capa_reglas
import fusion_entidades
import pipeline_url
import validador_spans
from extractor_articulo import extraer_desde_html, limpiar_boilerplate
from taxonomia_uaf import (
    NATURALEZA_POR_TIPO,
    naturaleza_de,
    normalizar_rol,
    normalizar_tipo,
    precedencia,
    tipo_desde_etiqueta_gliner,
)

TEXTO = (
    "El Juzgado de Garantía de Puente Alto formalizó a Ricardo Bustamante Leiva "
    "por lavado de activos. Las operaciones pasaron por Inversiones Bustamante y "
    "Compañía Limitada, RUT 76.543.210-3. La Unidad de Análisis Financiero remitió "
    "antecedentes al Ministerio Público. La comuna de San Ramón concentra los "
    "inmuebles. Sartor Administradora General de Fondos S.A. es distinta de Sartor "
    "Finance Group."
)


# ---------------------------------------------------------------------------
# Taxonomía
# ---------------------------------------------------------------------------


class PruebaTaxonomia(unittest.TestCase):
    def test_toda_persona_juridica_tiene_naturaleza_coherente(self):
        for tipo, natural in NATURALEZA_POR_TIPO.items():
            self.assertEqual(naturaleza_de(tipo), natural)

    def test_normalizacion_insensible_a_mayusculas_y_tildes(self):
        # Structured outputs no garantiza la capitalización de los enums.
        for variante in ("formalizado", "FORMALIZADO", "Formalizado"):
            rol, degradado = normalizar_rol(variante)
            self.assertEqual(rol, "FORMALIZADO")
            self.assertFalse(degradado)

    def test_rol_que_imputa_culpabilidad_se_anula(self):
        # No se suaviza a INVESTIGADO: eso seguiría afirmando una calidad
        # procesal cuya única fuente es el juicio del modelo que se descartó.
        for prohibido in ("CULPABLE", "lavador", "Testaferro", "narcotraficante"):
            rol, degradado = normalizar_rol(prohibido)
            self.assertEqual(rol, "SIN_ROL", f"{prohibido} debió anularse")
            self.assertTrue(degradado, f"{prohibido} debió marcarse como degradado")

    def test_rol_desconocido_no_revienta(self):
        rol, degradado = normalizar_rol("INVENTADO_POR_EL_MODELO")
        self.assertEqual(rol, "SIN_ROL")
        self.assertTrue(degradado)

    def test_tipo_desconocido_cae_en_otro(self):
        tipo, degradado = normalizar_tipo("SOCIEDAD_FANTASMA")
        self.assertEqual(tipo, "OTRO")
        self.assertTrue(degradado)

    def test_etiquetas_gliner_mapean_a_tipos_canonicos(self):
        self.assertEqual(tipo_desde_etiqueta_gliner("persona"), "PERSONA")
        self.assertEqual(
            tipo_desde_etiqueta_gliner("banco o institución financiera"),
            "INSTITUCION_FINANCIERA",
        )
        # Tolerancia a variación de capitalización devuelta por el modelo.
        self.assertEqual(tipo_desde_etiqueta_gliner("Persona"), "PERSONA")

    def test_regla_gana_sobre_llm_y_llm_sobre_gliner(self):
        self.assertGreater(precedencia("REGLA"), precedencia("LLM"))
        self.assertGreater(precedencia("LLM"), precedencia("GLINER"))


# ---------------------------------------------------------------------------
# Anclaje textual
# ---------------------------------------------------------------------------


class PruebaAnclaje(unittest.TestCase):
    def test_span_exacto_en_offset_declarado(self):
        inicio = TEXTO.index("Ricardo Bustamante Leiva")
        resultado = validador_spans.anclar(TEXTO, "Ricardo Bustamante Leiva", inicio)
        self.assertTrue(resultado.anclado)
        self.assertEqual(resultado.metodo, validador_spans.ANCLAJE_EXACTO)
        self.assertEqual(resultado.inicio, inicio)

    def test_offset_equivocado_se_reubica(self):
        # Los modelos cuentan caracteres mal; eso no es alucinación.
        resultado = validador_spans.anclar(TEXTO, "Ministerio Público", 9999)
        self.assertTrue(resultado.anclado)
        self.assertEqual(resultado.metodo, validador_spans.ANCLAJE_REUBICADO)
        self.assertEqual(
            TEXTO[resultado.inicio : resultado.fin], "Ministerio Público"
        )

    def test_tildes_alteradas_por_el_modelo_se_normalizan(self):
        resultado = validador_spans.anclar(TEXTO, "Ministerio Publico", None)
        self.assertTrue(resultado.anclado)
        self.assertEqual(resultado.metodo, validador_spans.ANCLAJE_NORMALIZADO)
        # Se conserva la superficie REAL del texto, no la del modelo.
        self.assertEqual(resultado.superficie, "Ministerio Público")

    def test_entidad_inexistente_se_rechaza(self):
        resultado = validador_spans.anclar(TEXTO, "Banco Falabella", 10)
        self.assertFalse(resultado.anclado)
        self.assertEqual(resultado.metodo, validador_spans.ANCLAJE_FALLIDO)

    def test_plegado_conserva_la_longitud(self):
        # Si el plegado cambiara el largo, los offsets se corromperían.
        for muestra in ("Ñuñoa", "Valparaíso", "José Ramón Ürzúa", TEXTO):
            self.assertEqual(
                len(validador_spans._plegar_conservando_longitud(muestra)), len(muestra)
            )

    def test_validar_lote_separa_ancladas_de_rechazadas(self):
        entidades = [
            {"span_exacto": "Ministerio Público", "offset_inicio": 0},
            {"span_exacto": "Corporación Inexistente S.A.", "offset_inicio": 5},
        ]
        aceptadas, rechazadas = validador_spans.validar_lote(TEXTO, entidades)
        self.assertEqual(len(aceptadas), 1)
        self.assertEqual(len(rechazadas), 1)
        self.assertIn("no aparece", rechazadas[0]["motivo_rechazo"])


# ---------------------------------------------------------------------------
# Capa de reglas
# ---------------------------------------------------------------------------


class PruebaReglas(unittest.TestCase):
    def test_modulo_11(self):
        self.assertEqual(capa_reglas.digito_verificador("76543210"), "3")
        self.assertTrue(capa_reglas.rut_valido("76.543.210-3"))
        self.assertFalse(capa_reglas.rut_valido("76.543.210-K"))

    def test_rut_invalido_no_se_emite(self):
        self.assertEqual(capa_reglas.detectar_ruts("RUT 76.543.210-K"), [])

    def test_rango_determina_naturaleza(self):
        ruts = capa_reglas.detectar_ruts("76.543.210-3 y 12.345.678-5")
        self.assertEqual(ruts[0]["naturaleza_por_rango"], "PERSONA_JURIDICA")
        self.assertEqual(ruts[1]["naturaleza_por_rango"], "PERSONA_NATURAL")

    def test_razon_social_con_antroponimo_es_persona_juridica(self):
        # El caso que rompe a los modelos NER clásicos.
        candidatos = capa_reglas.detectar(
            "Se investiga a Inversiones Bustamante y Compañía Limitada."
        )
        empresas = [c for c in candidatos if c.tipo == "EMPRESA"]
        self.assertTrue(empresas)
        self.assertIn("Bustamante", empresas[0].texto)

    def test_giro_declarado_prevalece_sobre_sufijo_generico(self):
        candidatos = capa_reglas.detectar("Sartor Administradora General de Fondos S.A.")
        tipos = {c.tipo for c in candidatos}
        self.assertIn("INSTITUCION_FINANCIERA", tipos)
        self.assertNotIn("EMPRESA", tipos)

    def test_encabezados_distinguen_tribunal_de_organismo(self):
        candidatos = {c.texto: c.tipo for c in capa_reglas.detectar(TEXTO)}
        self.assertEqual(candidatos.get("Juzgado de Garantía de Puente Alto"), "TRIBUNAL")
        self.assertEqual(candidatos.get("Ministerio Público"), "ORGANISMO_PUBLICO")
        self.assertEqual(
            candidatos.get("Unidad de Análisis Financiero"), "ORGANISMO_PUBLICO"
        )

    def test_rut_personal_no_se_asocia_a_persona_juridica(self):
        entidades = [
            {
                "inicio": 0,
                "fin": 10,
                "naturaleza": "PERSONA_JURIDICA",
                "texto": "Empresa X",
            }
        ]
        ruts = [
            {
                "rut": "12.345.678-5",
                "inicio": 15,
                "fin": 27,
                "naturaleza_por_rango": "PERSONA_NATURAL",
            }
        ]
        huerfanos = capa_reglas.asociar_ruts(entidades, ruts)
        self.assertEqual(len(huerfanos), 1)
        self.assertNotIn("ruts", entidades[0])


# ---------------------------------------------------------------------------
# Segmentación de GLiNER
# ---------------------------------------------------------------------------


class PruebaSegmentacion(unittest.TestCase):
    def test_texto_corto_no_se_segmenta(self):
        self.assertEqual(len(capa_gliner.segmentar("Texto breve.")), 1)

    def test_offsets_absolutos_reconstruyen_el_texto(self):
        largo = " ".join(
            f"La empresa Número {i} operó en la comuna de Ñuñoa durante el año 2026."
            for i in range(120)
        )
        segmentos = capa_gliner.segmentar(largo)
        self.assertGreater(len(segmentos), 1)
        for fragmento, offset in segmentos:
            # Sin esta igualdad, todo offset devuelto por GLiNER estaría corrido
            # y el validador de spans rechazaría entidades legítimas.
            self.assertEqual(largo[offset : offset + len(fragmento)], fragmento)

    def test_segmentacion_cubre_todo_el_texto(self):
        largo = "Oración de prueba número uno. " * 200
        segmentos = capa_gliner.segmentar(largo)
        cubierto = bytearray(len(largo))
        for fragmento, offset in segmentos:
            for i in range(offset, offset + len(fragmento)):
                cubierto[i] = 1
        self.assertEqual(sum(cubierto), len(largo), "Quedaron caracteres sin cubrir")

    def test_oracion_mas_larga_que_el_presupuesto_no_cuelga(self):
        # Prensa judicial produce oraciones de 300 palabras sin punto.
        sin_puntos = "palabra " * 2000
        segmentos = capa_gliner.segmentar(sin_puntos)
        self.assertGreater(len(segmentos), 1)
        for fragmento, _ in segmentos:
            self.assertLessEqual(len(fragmento), capa_gliner.CARACTERES_POR_SEGMENTO)

    def test_ausencia_de_gliner_no_lanza_excepcion(self):
        self.assertEqual(capa_gliner.detectar(""), [])


# ---------------------------------------------------------------------------
# Fusión y precedencia
# ---------------------------------------------------------------------------


class PruebaFusion(unittest.TestCase):
    def _registro(self, fuente, tipo, inicio=0, fin=10, **extra):
        base = {
            "texto": TEXTO[inicio:fin],
            "inicio": inicio,
            "fin": fin,
            "tipo": tipo,
            "fuente": fuente,
            "anclaje": validador_spans.ANCLAJE_EXACTO,
        }
        base.update(extra)
        return base

    def test_regla_prevalece_sobre_llm_en_conflicto(self):
        entidades = fusion_entidades.fusionar(
            [
                self._registro("REGLA", "EMPRESA"),
                self._registro("LLM", "PERSONA", confianza="ALTA"),
            ]
        )
        self.assertEqual(len(entidades), 1)
        self.assertEqual(entidades[0]["tipo"], "EMPRESA")
        self.assertEqual(entidades[0]["naturaleza"], "PERSONA_JURIDICA")

    def test_el_conflicto_queda_registrado_y_marca_validacion(self):
        entidades = fusion_entidades.fusionar(
            [
                self._registro("REGLA", "EMPRESA"),
                self._registro("LLM", "PERSONA", confianza="ALTA"),
            ]
        )
        self.assertTrue(entidades[0]["conflictos"])
        self.assertTrue(entidades[0]["requiere_validacion"])
        self.assertIn("conflicto", " ".join(entidades[0]["motivos_validacion"]))

    def test_acuerdo_entre_capas_sube_la_confianza(self):
        sola = fusion_entidades.fusionar([self._registro("LLM", "PERSONA", confianza="ALTA")])
        acompanada = fusion_entidades.fusionar(
            [
                self._registro("LLM", "PERSONA", confianza="ALTA"),
                self._registro("GLINER", "PERSONA", score=0.9),
            ]
        )
        self.assertGreater(acompanada[0]["confianza_score"], sola[0]["confianza_score"])

    def test_articulo_truncado_penaliza_la_confianza(self):
        completo = fusion_entidades.fusionar([self._registro("REGLA", "EMPRESA")], "COMPLETO")
        truncado = fusion_entidades.fusionar([self._registro("REGLA", "EMPRESA")], "PAYWALL")
        self.assertLess(truncado[0]["confianza_score"], completo[0]["confianza_score"])

    def test_anclaje_debil_penaliza_la_confianza(self):
        fuerte = fusion_entidades.fusionar([self._registro("LLM", "PERSONA", confianza="ALTA")])
        debil = fusion_entidades.fusionar(
            [
                self._registro(
                    "LLM", "PERSONA", confianza="ALTA",
                    anclaje=validador_spans.ANCLAJE_NORMALIZADO,
                )
            ]
        )
        self.assertLess(debil[0]["confianza_score"], fuerte[0]["confianza_score"])

    def test_rol_adverso_con_confianza_media_exige_revision(self):
        entidades = fusion_entidades.fusionar(
            [
                self._registro(
                    "LLM", "PERSONA", confianza="MEDIA", rol_procesal="FORMALIZADO"
                )
            ]
        )
        self.assertEqual(entidades[0]["rol_procesal"], "FORMALIZADO")
        self.assertTrue(entidades[0]["requiere_validacion"])

    def test_variantes_del_mismo_nombre_se_unifican(self):
        i1 = TEXTO.index("Ricardo Bustamante Leiva")
        entidades = fusion_entidades.fusionar(
            [
                self._registro(
                    "LLM", "PERSONA", i1, i1 + 24,
                    confianza="ALTA",
                    nombre_normalizado="Ricardo Bustamante Leiva",
                    variantes=["Ricardo Bustamante"],
                ),
                self._registro(
                    "LLM", "PERSONA", i1, i1 + 18,
                    confianza="ALTA",
                    nombre_normalizado="Ricardo Bustamante Leiva",
                    variantes=[],
                ),
            ]
        )
        self.assertEqual(len(entidades), 1)

    def test_marcas_compartidas_no_se_colapsan(self):
        # "Sartor AGF S.A." y "Sartor Finance Group" son personas jurídicas
        # distintas. Unificarlas por prefijo común sería un error sustantivo.
        i1 = TEXTO.index("Sartor Administradora General de Fondos S.A.")
        i2 = TEXTO.index("Sartor Finance Group")
        entidades = fusion_entidades.fusionar(
            [
                self._registro(
                    "REGLA", "INSTITUCION_FINANCIERA", i1, i1 + 43,
                    nombre_normalizado="Sartor Administradora General de Fondos S.A.",
                ),
                self._registro(
                    "LLM", "EMPRESA", i2, i2 + 20,
                    confianza="ALTA",
                    nombre_normalizado="Sartor Finance Group",
                    variantes=[],
                ),
            ]
        )
        self.assertEqual(len(entidades), 2)


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------


class PruebaExtraccion(unittest.TestCase):
    def test_boilerplate_chileno_se_elimina(self):
        sucio = (
            "El fiscal presentó la acusación.\n"
            "Lee también: Otro caso en Antofagasta\n"
            "Foto: Agencia Uno\n"
            "El tribunal resolvió el martes.\n"
            "Síguenos en redes sociales\n"
            "Etiquetas: judicial, fiscalía\n"
        )
        limpio = limpiar_boilerplate(sucio)
        self.assertIn("El fiscal presentó la acusación.", limpio)
        self.assertIn("El tribunal resolvió el martes.", limpio)
        for ruido in ("Lee también", "Agencia Uno", "Síguenos", "Etiquetas"):
            self.assertNotIn(ruido, limpio)

    def test_muro_de_pago_se_detecta(self):
        html = (
            "<html><body><article><p>"
            + "El caso comenzó en marzo pasado con una denuncia anónima. " * 3
            + "</p><p>Contenido exclusivo para suscriptores. Suscríbete para seguir leyendo.</p>"
            "</article></body></html>"
        )
        articulo = extraer_desde_html(html, url="https://ejemplo.cl/nota")
        self.assertEqual(articulo.estado_extraccion, "PAYWALL")
        self.assertTrue(articulo.advertencias)

    def test_html_vacio_no_revienta(self):
        articulo = extraer_desde_html("", url="https://ejemplo.cl/x")
        self.assertEqual(articulo.estado_extraccion, "VACIO")
        self.assertFalse(articulo.utilizable)

    def test_url_invalida_devuelve_error_controlado(self):
        resultado = pipeline_url.analizar_url("no-es-una-url", usar_llm=False, usar_gliner=False)
        self.assertEqual(resultado["articulo"]["estado_extraccion"], "ERROR")
        self.assertTrue(resultado["advertencias"])


# ---------------------------------------------------------------------------
# Pipeline completo con adjudicador simulado
# ---------------------------------------------------------------------------


def _respuesta_llm_falsa(texto, candidatos=None, metadatos=None, **kwargs):
    """Respuesta que mezcla salidas correctas con tres defectos deliberados."""
    return {
        "entidades": [
            {
                # Correcta.
                "span_exacto": "Ricardo Bustamante Leiva",
                "offset_inicio": texto.index("Ricardo Bustamante Leiva"),
                "nombre_normalizado": "Ricardo Bustamante Leiva",
                "tipo": "PERSONA",
                "rol_procesal": "FORMALIZADO",
                "evidencia": "formalizó a Ricardo Bustamante Leiva",
                "justificacion": "Nombre propio de persona con tratamiento procesal.",
                "confianza": "ALTA",
                "variantes": [],
            },
            {
                # Defecto 1: contradice a la capa de reglas.
                "span_exacto": "Inversiones Bustamante y Compañía Limitada",
                "offset_inicio": texto.index("Inversiones Bustamante"),
                "nombre_normalizado": "Inversiones Bustamante",
                "tipo": "PERSONA",
                "rol_procesal": "SIN_ROL",
                "evidencia": "Las operaciones pasaron por Inversiones Bustamante",
                "justificacion": "Contiene un apellido.",
                "confianza": "ALTA",
                "variantes": [],
            },
            {
                # Defecto 2: alucinación pura.
                "span_exacto": "Banco Internacional de Panamá",
                "offset_inicio": 40,
                "nombre_normalizado": "Banco Internacional de Panamá",
                "tipo": "INSTITUCION_FINANCIERA",
                "rol_procesal": "INVESTIGADO",
                "evidencia": "el banco panameño recibió los fondos",
                "justificacion": "Mencionado en el artículo.",
                "confianza": "MEDIA",
                "variantes": [],
            },
            {
                # Defecto 3: rol que imputa culpabilidad.
                "span_exacto": "San Ramón",
                "offset_inicio": texto.index("San Ramón"),
                "nombre_normalizado": "San Ramón",
                "tipo": "LUGAR",
                "rol_procesal": "CULPABLE",
                "evidencia": "La comuna de San Ramón concentra los inmuebles",
                "justificacion": "Topónimo comunal.",
                "confianza": "ALTA",
                "variantes": [],
            },
        ],
        "relaciones": [
            {
                "origen": "Ricardo Bustamante Leiva",
                "tipo_relacion": "CONTROLA_A",
                "destino": "Inversiones Bustamante y Compañía Limitada",
                "evidencia": "Las operaciones pasaron por Inversiones Bustamante",
            },
            {
                # Evidencia inventada: debe descartarse.
                "origen": "Ricardo Bustamante Leiva",
                "tipo_relacion": "TRANSFIERE_A",
                "destino": "Banco Internacional de Panamá",
                "evidencia": "giró los fondos al extranjero en diciembre",
            },
        ],
        "delitos_mencionados": ["lavado de activos"],
        "sintesis": "Formalización por lavado de activos.",
        "hay_procedimiento": True,
        "_simulado": False,
        "_uso": {"input_tokens": 1200, "output_tokens": 400},
        "_modelo": "prueba",
    }


class PruebaPipelineCompleto(unittest.TestCase):
    def setUp(self):
        parche = mock.patch.object(pipeline_url, "adjudicar", _respuesta_llm_falsa)
        parche.start()
        self.addCleanup(parche.stop)
        self.resultado = pipeline_url.analizar_texto(TEXTO, usar_gliner=False)
        self.por_texto = {e["texto"]: e for e in self.resultado["entidades"]}

    def test_la_alucinacion_no_llega_a_la_salida(self):
        nombres = " ".join(self.por_texto)
        self.assertNotIn("Panamá", nombres)
        self.assertEqual(self.resultado["capas"]["L3_adjudicador"]["rechazadas_por_anclaje"], 1)

    def test_la_regla_gana_al_modelo_sobre_la_razon_social(self):
        entidad = self.por_texto["Inversiones Bustamante y Compañía Limitada"]
        self.assertEqual(entidad["naturaleza"], "PERSONA_JURIDICA")
        self.assertEqual(entidad["tipo"], "EMPRESA")
        self.assertTrue(entidad["conflictos"])
        self.assertTrue(entidad["requiere_validacion"])

    def test_el_rol_prohibido_se_anula_y_marca_revision(self):
        entidad = self.por_texto["San Ramón"]
        self.assertNotEqual(entidad["rol_procesal"], "CULPABLE")
        self.assertEqual(entidad["rol_procesal"], "SIN_ROL")
        self.assertTrue(entidad["requiere_validacion"])
        self.assertIn("no era admisible", " ".join(entidad["motivos_validacion"]))

    def test_la_evidencia_sobrevive_a_la_anulacion_del_rol(self):
        # El rol se anula, pero la frase del artículo se conserva para que el
        # analista decida por sí mismo qué calidad procesal corresponde.
        entidad = self.por_texto["San Ramón"]
        self.assertTrue(entidad["evidencia"])
        self.assertIn(entidad["evidencia"], TEXTO)

    def test_rol_invalido_impide_confianza_maxima(self):
        entidad = self.por_texto["San Ramón"]
        self.assertLess(
            entidad["confianza_score"],
            1.0,
            "Una entidad marcada para revisión no puede salir con confianza máxima.",
        )

    def test_el_rol_procesal_legitimo_se_conserva(self):
        entidad = self.por_texto["Ricardo Bustamante Leiva"]
        self.assertEqual(entidad["rol_procesal"], "FORMALIZADO")
        self.assertEqual(entidad["tipo"], "PERSONA")
        self.assertIn("LLM", entidad["fuentes"])

    def test_relacion_con_evidencia_inventada_se_descarta(self):
        tipos = {r["tipo_relacion"] for r in self.resultado["relaciones"]}
        self.assertIn("CONTROLA_A", tipos)
        self.assertNotIn("TRANSFIERE_A", tipos)

    def test_la_salida_declara_que_capas_corrieron(self):
        capas = self.resultado["capas"]
        self.assertTrue(capas["L1_reglas"]["ejecutada"])
        self.assertFalse(capas["L2_gliner"]["ejecutada"])
        self.assertTrue(capas["L3_adjudicador"]["ejecutada"])
        self.assertTrue(capas["L4_fusion"]["ejecutada"])

    def test_toda_entidad_es_verificable_en_el_texto(self):
        # La propiedad central del pipeline: no se emite nada que no esté ahí.
        for entidad in self.resultado["entidades"]:
            self.assertEqual(
                TEXTO[entidad["inicio"] : entidad["fin"]],
                entidad["texto"],
                f"Span no verificable: {entidad['texto']}",
            )


class PruebaDegradacion(unittest.TestCase):
    def test_sin_credencial_el_pipeline_sigue_con_reglas(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
            resultado = pipeline_url.analizar_texto(TEXTO, usar_gliner=False)
        self.assertFalse(resultado["capas"]["L3_adjudicador"]["ejecutada"])
        self.assertTrue(resultado["entidades"], "L1 debe seguir produciendo entidades")

    def test_sin_ninguna_capa_opcional_no_revienta(self):
        resultado = pipeline_url.analizar_texto(TEXTO, usar_gliner=False, usar_llm=False)
        self.assertGreater(len(resultado["entidades"]), 3)
        self.assertEqual(resultado["resumen"]["total"], len(resultado["entidades"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class PruebaRolImpropio(unittest.TestCase):
    """Un lugar no puede ser imputado: no es sujeto de derecho."""

    def test_lugar_no_conserva_rol_procesal(self):
        i = TEXTO.index("San Ramón")
        entidades = fusion_entidades.fusionar(
            [
                {
                    "texto": "San Ramón",
                    "inicio": i,
                    "fin": i + 9,
                    "tipo": "LUGAR",
                    "fuente": "LLM",
                    "confianza": "ALTA",
                    "rol_procesal": "IMPUTADO",
                    "anclaje": validador_spans.ANCLAJE_EXACTO,
                }
            ]
        )
        self.assertEqual(entidades[0]["rol_procesal"], "SIN_ROL")
        self.assertTrue(entidades[0]["requiere_validacion"])
        self.assertTrue(any("sujeto de derecho" in c for c in entidades[0]["conflictos"]))


# ---------------------------------------------------------------------------
# Adaptador GLiNER2
# ---------------------------------------------------------------------------


class PruebaGliner2(unittest.TestCase):
    """El aplanado de la salida agrupada y el remapeo de offsets."""

    def _salida_falsa(self, fragmento):
        """Imita el formato real: agrupado por etiqueta, con spans."""
        i = fragmento.find("Ricardo Bustamante Leiva")
        j = fragmento.find("Ministerio Público")
        entidades = {}
        if i >= 0:
            entidades["persona"] = [
                {"text": "Ricardo Bustamante Leiva", "start": i, "end": i + 24,
                 "confidence": 0.91}
            ]
        if j >= 0:
            entidades["organismo_publico"] = [
                {"text": "Ministerio Público", "start": j, "end": j + 18,
                 "confidence": 0.88}
            ]
        return {"entities": entidades}

    def _motor(self):
        motor = mock.MagicMock()
        motor.extract_entities.side_effect = (
            lambda frag, esquema, **kw: self._salida_falsa(frag)
        )
        return motor

    def test_aplana_la_salida_agrupada_por_etiqueta(self):
        with mock.patch.object(capa_gliner2, "cargar_modelo", return_value=self._motor()):
            candidatos = capa_gliner2.detectar(TEXTO)
        tipos = {c.tipo for c in candidatos}
        self.assertIn("PERSONA", tipos)
        self.assertIn("ORGANISMO_PUBLICO", tipos)

    def test_los_offsets_apuntan_al_texto_original(self):
        with mock.patch.object(capa_gliner2, "cargar_modelo", return_value=self._motor()):
            candidatos = capa_gliner2.detectar(TEXTO)
        self.assertTrue(candidatos)
        for cand in candidatos:
            self.assertEqual(TEXTO[cand.inicio:cand.fin], cand.texto)

    def test_exige_include_spans(self):
        # Sin offsets el pipeline completo deja de funcionar, así que la
        # llamada debe pedirlos siempre, no dejarlo al valor por defecto.
        motor = self._motor()
        with mock.patch.object(capa_gliner2, "cargar_modelo", return_value=motor):
            capa_gliner2.detectar(TEXTO)
        for llamada in motor.extract_entities.call_args_list:
            self.assertIs(llamada.kwargs.get("include_spans"), True)
            self.assertIs(llamada.kwargs.get("include_confidence"), True)

    def test_descarta_candidatos_sin_offset(self):
        motor = mock.MagicMock()
        motor.extract_entities.return_value = {
            "entities": {"persona": [{"text": "Alguien"}]}   # sin start/end
        }
        with mock.patch.object(capa_gliner2, "cargar_modelo", return_value=motor):
            self.assertEqual(capa_gliner2.detectar(TEXTO), [])

    def test_descarta_offsets_corridos(self):
        motor = mock.MagicMock()
        motor.extract_entities.return_value = {
            "entities": {"persona": [
                {"text": "Ricardo Bustamante Leiva", "start": 0, "end": 24,
                 "confidence": 0.9}
            ]}
        }
        with mock.patch.object(capa_gliner2, "cargar_modelo", return_value=motor):
            candidatos = capa_gliner2.detectar(TEXTO)
        # El texto en la posición 0 no es ese nombre: el candidato se rechaza.
        self.assertEqual(candidatos, [])

    def test_esquema_cubre_toda_la_taxonomia_juridica(self):
        from taxonomia_uaf import TIPOS_PERSONA_JURIDICA
        cubiertos = {capa_gliner2.tipo_desde_etiqueta(e) for e in capa_gliner2.ESQUEMA_UAF}
        faltan = TIPOS_PERSONA_JURIDICA - cubiertos
        self.assertFalse(faltan, f"Sin etiqueta GLiNER2: {faltan}")

    def test_modelo_por_defecto_es_multilingue(self):
        # El modelo del README oficial es solo inglés; sobre prensa chilena
        # rinde mal. Esta prueba impide que se cuele por descuido.
        self.assertIn("multi", capa_gliner2.MODELO_POR_DEFECTO.lower())
