#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capa L3 — Adjudicación semántica con la API de Claude.

Qué hace y qué no
-----------------
El adjudicador **no busca entidades desde cero**. Recibe el texto limpio y la
lista de candidatos que produjeron las capas L1 (reglas) y L2 (GLiNER), y
resuelve las cuatro cosas que ninguna de las dos puede resolver:

1. Naturaleza jurídica cuando no hay marcador léxico
   ("Sartor" como apellido frente a "Sartor" como grupo empresarial).
2. Correferencia entre variantes del mismo referente
   ("Marcela Ortiz" y "Marcela Ortiz Vega" son la misma; "Sartor Finance Group"
   y "Sartor AGF S.A." no lo son).
3. **Rol procesal**, que es donde está el valor analítico. Una entidad sin rol
   no sirve para adverse media screening; entidad + rol + delito + fecha sí.
4. Relaciones explícitas entre entidades, con la frase que las sustenta.

Garantías técnicas
------------------
- ``output_config.format`` con JSON Schema: el decoding queda restringido por
  gramática, de modo que la respuesta siempre parsea. No hay lógica de reintento
  por JSON malformado porque no puede haber JSON malformado.
- Todos los campos del esquema son obligatorios. Además de simplificar el
  parseo, evita el reordenamiento de propiedades opcionales y mantiene la
  complejidad del esquema muy por debajo del límite de compilación.
- El bloque estático del prompt (taxonomía, reglas, ejemplos) lleva
  ``cache_control``. Se reutiliza entre artículos: en una corrida de 200
  noticias solo se paga completo la primera vez.
- Sin dependencias externas. Se llama la API REST con ``urllib``, igual que el
  Monitor Legislativo, para no agregar paquetes al workflow.

La verificación de que las entidades existen en el texto **no ocurre aquí**.
Ocurre en ``validador_spans.py``, sobre la respuesta ya recibida.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from taxonomia_uaf import ROLES_PROCESALES, TIPOS_ADJUDICABLES

VERSION_ADJUDICADOR = "1.0.0"

URL_API = "https://api.anthropic.com/v1/messages"
VERSION_API = "2023-06-01"

#: Sonnet da el mejor equilibrio para desambiguación semántica en español.
#: Para corridas masivas del monitor, 'claude-haiku-4-5' baja el costo de forma
#: sustantiva; conviene medir la caída de F1 sobre el gold standard antes de
#: cambiar.
MODELO_POR_DEFECTO = os.environ.get("CLAUDE_MODELO", "claude-sonnet-4-6")

MAX_TOKENS = 8000
REINTENTOS = 3


TIPOS_RELACION: tuple[str, ...] = (
    "CONTROLA_A",
    "BENEFICIARIO_FINAL_DE",
    "REPRESENTA_A",
    "SOCIO_DE",
    "DIRECTOR_DE",
    "TRABAJA_EN",
    "INVESTIGA_A",
    "FORMALIZA_A",
    "SANCIONA_A",
    "REPORTA_A",
    "DEFIENDE_A",
    "ABSUELVE_A",
    "CONDENA_A",
    "TRANSFIERE_A",
    "VINCULADO_A",
)


# ---------------------------------------------------------------------------
# Esquema de salida
# ---------------------------------------------------------------------------


def construir_esquema() -> dict[str, Any]:
    """JSON Schema para ``output_config.format``.

    Restricciones respetadas a propósito: sin propiedades opcionales, sin tipos
    unión (``anyOf`` o ``["string","null"]``), ``additionalProperties: false``
    en todos los objetos. La ausencia de valor se representa con cadena vacía o
    arreglo vacío, no con ``null``.
    """
    return {
        "type": "object",
        "properties": {
            "entidades": {
                "type": "array",
                "description": (
                    "Entidades nombradas presentes en el artículo. Cada una debe "
                    "existir literalmente en el texto."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "span_exacto": {
                            "type": "string",
                            "description": (
                                "La cadena copiada CARACTER POR CARACTER del texto, "
                                "con sus tildes y mayúsculas originales. No la "
                                "corrijas ni la completes."
                            ),
                        },
                        "offset_inicio": {
                            "type": "integer",
                            "description": (
                                "Posición del primer carácter de span_exacto dentro "
                                "del texto. Si no la puedes calcular con certeza, "
                                "entrega tu mejor estimación: se reubica luego."
                            ),
                        },
                        "nombre_normalizado": {
                            "type": "string",
                            "description": (
                                "Forma canónica del referente, usada para unificar "
                                "variantes. Para personas, el nombre más completo "
                                "que aparezca en el artículo."
                            ),
                        },
                        "tipo": {
                            "type": "string",
                            "enum": sorted(TIPOS_ADJUDICABLES),
                        },
                        "rol_procesal": {
                            "type": "string",
                            "enum": list(ROLES_PROCESALES),
                        },
                        "evidencia": {
                            "type": "string",
                            "description": (
                                "Fragmento LITERAL del artículo, de una oración como "
                                "máximo, que sustenta el tipo y el rol asignados. "
                                "Debe poder encontrarse con búsqueda de texto."
                            ),
                        },
                        "justificacion": {
                            "type": "string",
                            "description": (
                                "Una frase explicando la clasificación. Ej: 'razón "
                                "social con apellido, el sufijo Ltda. la define como "
                                "sociedad'."
                            ),
                        },
                        "confianza": {
                            "type": "string",
                            "enum": ["ALTA", "MEDIA", "BAJA"],
                        },
                        "variantes": {
                            "type": "array",
                            "description": (
                                "Otras cadenas del mismo artículo que designan a este "
                                "mismo referente. Arreglo vacío si no hay."
                            ),
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "span_exacto",
                        "offset_inicio",
                        "nombre_normalizado",
                        "tipo",
                        "rol_procesal",
                        "evidencia",
                        "justificacion",
                        "confianza",
                        "variantes",
                    ],
                    "additionalProperties": False,
                },
            },
            "relaciones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "origen": {"type": "string"},
                        "tipo_relacion": {"type": "string", "enum": list(TIPOS_RELACION)},
                        "destino": {"type": "string"},
                        "evidencia": {
                            "type": "string",
                            "description": "Fragmento literal que sustenta la relación.",
                        },
                    },
                    "required": ["origen", "tipo_relacion", "destino", "evidencia"],
                    "additionalProperties": False,
                },
            },
            "delitos_mencionados": {
                "type": "array",
                "description": (
                    "Figuras penales o infraccionales nombradas en el artículo, tal "
                    "como aparecen. No las infieras."
                ),
                "items": {"type": "string"},
            },
            "sintesis": {
                "type": "string",
                "description": "Dos frases sobre el hecho que reporta el artículo.",
            },
            "hay_procedimiento": {
                "type": "boolean",
                "description": (
                    "Verdadero si el artículo reporta una investigación, proceso "
                    "judicial o sanción administrativa en curso o resuelta."
                ),
            },
        },
        "required": [
            "entidades",
            "relaciones",
            "delitos_mencionados",
            "sintesis",
            "hay_procedimiento",
        ],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

INSTRUCCIONES = """\
Eres un componente de extracción de un sistema de inteligencia financiera que \
analiza prensa chilena. Tu salida alimenta a analistas humanos; no toma \
decisiones por sí sola.

TAREA
Recibes el texto de un artículo y una lista de candidatos detectados por capas \
previas. Debes devolver las entidades nombradas del artículo, clasificadas, con \
su rol procesal y las relaciones explícitas entre ellas.

Los candidatos son una ayuda, no una restricción: puedes descartar los que sean \
ruido y agregar entidades que las capas previas no vieron.

REGLA DE ANCLAJE (la más importante)
Cada entidad debe traer en `span_exacto` una cadena copiada literalmente del \
texto. Un proceso posterior verifica con comparación de strings que esa cadena \
esté en el artículo y descarta todo lo que no lo esté. No hay ningún beneficio \
en proponer algo que no aparece: se pierde. Lo mismo con `evidencia`, que debe \
ser un fragmento literal.

Si no estás seguro de una entidad, inclúyela con `confianza: BAJA` en vez de \
omitirla. La revisión humana filtra; el olvido no se recupera.

CRITERIOS DE CLASIFICACIÓN (Chile)

- El sufijo societario manda sobre todo lo demás. "Inversiones Antonio Ramírez y \
  Compañía Limitada" es EMPRESA, no PERSONA, aunque contenga un nombre propio.
- Muchas comunas chilenas llevan nombre de persona: Pedro Aguirre Cerda, San \
  Ramón, Padre Hurtado, Diego de Almagro. En contexto territorial son LUGAR.
- Distingue el órgano del Estado (ORGANISMO_PUBLICO) del tribunal (TRIBUNAL) y \
  de la empresa privada (EMPRESA). Fiscalía, Contraloría, CMF, SII, UAF, PDI son \
  ORGANISMO_PUBLICO. Corte de Apelaciones y Juzgado de Garantía son TRIBUNAL.
- Bancos, cooperativas de ahorro y crédito, corredoras de bolsa, AGF y AFP son \
  INSTITUCION_FINANCIERA.
- Fundaciones, corporaciones y universidades son ENTIDAD_SIN_FINES_DE_LUCRO. \
  Asociaciones gremiales, sindicatos y partidos son ORGANIZACION.
- Cargos y profesiones no son entidades. "el fiscal" no lo es; "el fiscal Carlos \
  Palma" produce la entidad "Carlos Palma".
- Nombres de leyes, causas y operaciones policiales no son entidades.

CORREFERENCIA
Unifica variantes del mismo referente en `variantes`: "Marcela Ortiz" y \
"Marcela Ortiz Vega" son una sola entidad. NO unifiques personas jurídicas \
distintas que compartan marca: "Sartor Finance Group" y "Sartor Administradora \
General de Fondos S.A." son entidades separadas, porque tienen personalidad \
jurídica separada y esa distinción es sustantiva para el análisis.

ROL PROCESAL — RESTRICCIÓN LEGAL
Asigna el rol que el artículo afirma, nunca el que se pueda inferir.
- Usa FORMALIZADO solo si el texto dice que hubo formalización.
- Usa CONDENADO solo si hay condena; ABSUELTO y SOBRESEIDO igual.
- Ser investigado, imputado o formalizado NO es ser responsable. No existe un \
  rol que atribuya culpabilidad y no debes crear uno.
- Si la persona aparece sin vínculo con un procedimiento, usa SIN_ROL o \
  MENCIONADO.
- El organismo que investiga lleva FISCALIZADOR, no un rol adverso.

Ante duda entre un rol adverso y uno neutro, elige el neutro y baja la \
confianza. El costo de sub-clasificar es una revisión; el de sobre-clasificar \
es imputar a alguien un estado procesal que no tiene.

RELACIONES
Solo relaciones afirmadas explícitamente en el texto, con su frase de sustento. \
No infieras control societario ni beneficiario final a partir de coincidencia \
de apellidos.
"""


def construir_mensaje_usuario(
    texto: str, candidatos: list[dict[str, Any]], metadatos: dict[str, Any]
) -> str:
    lineas = [
        "<metadatos_articulo>",
        f"medio: {metadatos.get('medio', '')}",
        f"titulo: {metadatos.get('titulo', '')}",
        f"fecha: {metadatos.get('fecha_publicacion', '')}",
        f"estado_extraccion: {metadatos.get('estado_extraccion', '')}",
        "</metadatos_articulo>",
        "",
        "<candidatos_capas_previas>",
    ]
    if candidatos:
        for cand in candidatos[:200]:
            lineas.append(
                f"- «{cand.get('texto', '')}» "
                f"[offset {cand.get('inicio', '?')}] "
                f"tipo_propuesto={cand.get('tipo', '?')} "
                f"fuente={cand.get('fuente', '?')}"
            )
    else:
        lineas.append("(ninguno: las capas previas no produjeron candidatos)")
    lineas += [
        "</candidatos_capas_previas>",
        "",
        "<texto_articulo>",
        texto,
        "</texto_articulo>",
        "",
        "Devuelve el análisis conforme al esquema.",
    ]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Llamada a la API
# ---------------------------------------------------------------------------


class ErrorAdjudicador(Exception):
    pass


def adjudicar(
    texto: str,
    candidatos: list[dict[str, Any]] | None = None,
    metadatos: dict[str, Any] | None = None,
    modelo: str = MODELO_POR_DEFECTO,
    api_key: str = "",
    simular: bool = False,
) -> dict[str, Any]:
    """Ejecuta la adjudicación. Devuelve el dict validado por esquema.

    En modo ``simular`` no llama a la API y devuelve una estructura vacía
    válida, para poder ejercitar el pipeline completo sin credenciales.
    """
    candidatos = candidatos or []
    metadatos = metadatos or {}

    if simular:
        return {
            "entidades": [],
            "relaciones": [],
            "delitos_mencionados": [],
            "sintesis": "",
            "hay_procedimiento": False,
            "_simulado": True,
            "_uso": {},
        }

    clave = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not clave:
        raise ErrorAdjudicador(
            "Falta la variable de entorno ANTHROPIC_API_KEY. En GitHub Actions se "
            "define como secreto del repositorio."
        )

    cuerpo = {
        "model": modelo,
        "max_tokens": MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": INSTRUCCIONES,
                # El bloque de instrucciones es idéntico para todos los
                # artículos. Marcarlo como cacheable evita repagarlo en cada
                # noticia de la corrida.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": construir_mensaje_usuario(texto, candidatos, metadatos),
            }
        ],
        "output_config": {
            "format": {"type": "json_schema", "schema": construir_esquema()}
        },
    }

    respuesta = _post_con_reintentos(cuerpo, clave)

    motivo = respuesta.get("stop_reason", "")
    if motivo == "refusal":
        raise ErrorAdjudicador(
            "El modelo declinó procesar el artículo. Revise el contenido antes de "
            "reintentar."
        )
    if motivo == "max_tokens":
        raise ErrorAdjudicador(
            "La respuesta se truncó por límite de tokens; el JSON está incompleto. "
            f"Aumente MAX_TOKENS (actual: {MAX_TOKENS}) o divida el artículo."
        )

    bloques = [b for b in respuesta.get("content", []) if b.get("type") == "text"]
    if not bloques:
        raise ErrorAdjudicador("La respuesta no contiene bloques de texto.")

    try:
        datos = json.loads(bloques[0]["text"])
    except json.JSONDecodeError as exc:  # pragma: no cover - no debería ocurrir
        raise ErrorAdjudicador(
            f"JSON inválido pese a structured outputs: {exc}"
        ) from exc

    datos["_simulado"] = False
    datos["_uso"] = respuesta.get("usage", {})
    datos["_modelo"] = respuesta.get("model", modelo)
    return datos


def _post_con_reintentos(cuerpo: dict[str, Any], clave: str) -> dict[str, Any]:
    datos = json.dumps(cuerpo).encode("utf-8")
    peticion = urllib.request.Request(
        URL_API,
        data=datos,
        headers={
            "content-type": "application/json",
            "x-api-key": clave,
            "anthropic-version": VERSION_API,
        },
        method="POST",
    )

    ultimo = ""
    for intento in range(REINTENTOS):
        try:
            with urllib.request.urlopen(peticion, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detalle = exc.read().decode("utf-8", errors="replace")[:400]
            ultimo = f"HTTP {exc.code}: {detalle}"
            # 429 y 5xx son transitorios; 4xx restantes no se reintentan.
            if exc.code not in (429, 500, 502, 503, 529):
                raise ErrorAdjudicador(ultimo) from exc
        except Exception as exc:
            ultimo = str(exc)

        if intento < REINTENTOS - 1:
            time.sleep(2**intento * 2)

    raise ErrorAdjudicador(f"Fallo tras {REINTENTOS} intentos. Último error: {ultimo}")


def diagnostico() -> dict[str, Any]:
    return {
        "version": VERSION_ADJUDICADOR,
        "modelo": MODELO_POR_DEFECTO,
        "api_key_presente": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "n_tipos_adjudicables": len(TIPOS_ADJUDICABLES),
        "n_roles": len(ROLES_PROCESALES),
        "n_tipos_relacion": len(TIPOS_RELACION),
    }
