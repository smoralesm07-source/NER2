#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capa L0 — Extracción del artículo a partir de una URL.

Por qué esta capa importa más de lo que parece
----------------------------------------------
En prensa chilena, el cuerpo del artículo viene envuelto en bloques que un
extractor ingenuo arrastra al texto: menú de navegación, "Lee también",
"Te puede interesar", créditos de foto, pie de suscripción y, en varios medios,
un carrusel de titulares relacionados *dentro* del cuerpo. Cada uno de esos
bloques inyecta entidades que no pertenecen al artículo. Ninguna capa posterior
puede distinguir un falso positivo de extracción de una entidad legítima,
porque el texto que recibe ya está contaminado.

Estrategia
----------
1. ``trafilatura`` como extractor principal. Es la mejor librería open source
   para boilerplate removal y además devuelve metadatos (título, fecha, autor,
   medio).
2. Respaldo con la biblioteca estándar si trafilatura no está instalada, para
   que el módulo nunca haga caer el workflow.
3. Barrido posterior de bloques residuales conocidos de medios chilenos.
4. Diagnóstico explícito del estado: COMPLETO / TRUNCADO / PAYWALL / VACIO.
   Un artículo truncado no debe procesarse con la misma confianza que uno
   completo; el estado viaja hasta la salida final.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any

VERSION_EXTRACTOR = "1.0.0"

try:
    import trafilatura
    from trafilatura.settings import use_config as _tf_use_config

    TRAFILATURA_DISPONIBLE = True
except Exception:  # pragma: no cover - degradación controlada
    trafilatura = None
    _tf_use_config = None
    TRAFILATURA_DISPONIBLE = False


USER_AGENT = (
    "Mozilla/5.0 (compatible; MonitorUAF/1.0; +analisis-de-prensa) "
    "Python-urllib"
)

#: Bajo este umbral de caracteres el cuerpo no es un artículo utilizable.
MINIMO_CARACTERES = 280

#: Entre MINIMO y este umbral, el texto se marca TRUNCADO: suficiente para
#: analizar pero probablemente un extracto o un lead cortado por muro de pago.
UMBRAL_TRUNCADO = 900


# ---------------------------------------------------------------------------
# Bloques residuales de medios chilenos
# ---------------------------------------------------------------------------
#
# Se aplican DESPUÉS de trafilatura, sobre el texto ya limpio, porque son
# fragmentos que sobreviven al boilerplate removal genérico al estar dentro
# del contenedor del artículo.

_PATRONES_BOILERPLATE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"^\s*(lee|revisa|mira|te puede interesar|también te puede interesar)\b.*$",
        r"^\s*(lee también|leer también|lea también|ver también)\b.*$",
        r"^\s*(síguenos|siguenos|suscríbete|suscribete|comparte|compartir)\b.*$",
        r"^\s*(foto|fotos|imagen|crédito|credito|créditos)\s*:.*$",
        r"^\s*(agencia\s+uno|aton\s+chile|photosport)\b.*$",
        r"^\s*(publicidad|contenido patrocinado|advertorial)\s*$",
        r"^\s*(etiquetas|tags)\s*:.*$",
        r"^\s*(comentarios|deja tu comentario)\b.*$",
        r"^\s*(newsletter|recibe las noticias)\b.*$",
        r"^\s*(all rights reserved|todos los derechos reservados)\b.*$",
    )
)

#: Marcadores textuales de muro de pago. Su presencia junto a un cuerpo corto
#: es evidencia suficiente para declarar PAYWALL.
_MARCADORES_PAYWALL: tuple[str, ...] = (
    "contenido exclusivo para suscriptores",
    "exclusivo suscriptores",
    "suscribete para seguir leyendo",
    "suscribase para seguir leyendo",
    "registrate para seguir leyendo",
    "para continuar leyendo",
    "ya eres suscriptor",
    "hazte suscriptor",
    "este articulo es exclusivo",
    "inicia sesion para leer",
)


@dataclass
class ArticuloExtraido:
    """Resultado de la capa L0."""

    url: str
    texto: str = ""
    titulo: str = ""
    bajada: str = ""
    fecha_publicacion: str = ""
    autor: str = ""
    medio: str = ""
    estado_extraccion: str = "VACIO"  # COMPLETO | TRUNCADO | PAYWALL | VACIO | ERROR
    n_caracteres: int = 0
    extractor: str = ""
    advertencias: list[str] = field(default_factory=list)

    @property
    def utilizable(self) -> bool:
        return self.estado_extraccion in {"COMPLETO", "TRUNCADO"}

    #: El texto que las capas L1/L2/L3 deben usar. Título y bajada se
    #: anteponen porque concentran las entidades principales y trafilatura los
    #: entrega por separado del cuerpo.
    @property
    def texto_analizable(self) -> str:
        partes = [p.strip() for p in (self.titulo, self.bajada, self.texto) if p and p.strip()]
        return "\n\n".join(partes)

    def a_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["utilizable"] = self.utilizable
        return d


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------


def descargar_html(url: str, timeout: int = 20) -> tuple[str, str]:
    """Devuelve ``(html, error)``. Nunca lanza excepción."""
    if not re.match(r"^https?://", url or "", re.IGNORECASE):
        return "", "La URL debe comenzar con http:// o https://"

    peticion = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "es-CL,es;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
            crudo = respuesta.read()
            codificacion = respuesta.headers.get_content_charset() or "utf-8"
        return crudo.decode(codificacion, errors="replace"), ""
    except urllib.error.HTTPError as exc:
        return "", f"HTTP {exc.code} al solicitar la URL"
    except urllib.error.URLError as exc:
        return "", f"No se pudo conectar: {exc.reason}"
    except Exception as exc:  # pragma: no cover
        return "", f"Fallo inesperado en la descarga: {exc}"


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------


def extraer_desde_html(html: str, url: str = "") -> ArticuloExtraido:
    """Extrae cuerpo y metadatos de un HTML ya descargado."""
    articulo = ArticuloExtraido(url=url)

    if not html or not html.strip():
        articulo.estado_extraccion = "VACIO"
        articulo.advertencias.append("El HTML recibido está vacío.")
        return articulo

    if TRAFILATURA_DISPONIBLE:
        articulo.extractor = f"trafilatura/{getattr(trafilatura, '__version__', '?')}"
        _extraer_con_trafilatura(html, url, articulo)
    else:
        articulo.extractor = "respaldo-stdlib"
        articulo.advertencias.append(
            "trafilatura no está instalada; se usó el extractor de respaldo, "
            "con mayor riesgo de arrastrar boilerplate."
        )
        _extraer_con_respaldo(html, articulo)

    articulo.texto = limpiar_boilerplate(articulo.texto)
    articulo.n_caracteres = len(articulo.texto_analizable)
    articulo.estado_extraccion = _diagnosticar(html, articulo)
    return articulo


def _extraer_con_trafilatura(html: str, url: str, articulo: ArticuloExtraido) -> None:
    config = _tf_use_config()
    # Sin esto trafilatura intenta llamadas de red adicionales para metadatos,
    # lo que en un runner sin salida a internet cuelga el job.
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

    articulo.texto = (
        trafilatura.extract(
            html,
            url=url or None,
            favor_precision=True,       # precisión sobre recall: es lo que queremos
            include_comments=False,     # los comentarios son fuente masiva de ruido
            include_tables=False,
            include_images=False,
            include_links=False,
            deduplicate=True,
            config=config,
        )
        or ""
    )

    try:
        meta = trafilatura.extract_metadata(html, default_url=url or None)
    except Exception:
        meta = None

    if meta is not None:
        articulo.titulo = _texto(getattr(meta, "title", ""))
        articulo.bajada = _texto(getattr(meta, "description", ""))
        articulo.fecha_publicacion = _texto(getattr(meta, "date", ""))
        articulo.medio = _texto(getattr(meta, "sitename", "")) or _texto(
            getattr(meta, "hostname", "")
        )
        autor = getattr(meta, "author", "")
        articulo.autor = _texto(autor if isinstance(autor, str) else "; ".join(autor or []))

    # La bajada solo aporta si no es un recorte literal del cuerpo.
    if articulo.bajada and articulo.bajada[:60] in articulo.texto[:400]:
        articulo.bajada = ""


class _RecolectorTexto:
    """Respaldo mínimo sin dependencias: extrae texto de <p> del documento."""

    def __init__(self) -> None:
        from html.parser import HTMLParser

        recolector = self

        class _Parser(HTMLParser):
            def __init__(self) -> None:
                super().__init__(convert_charrefs=True)
                self.ignorar = 0
                self.en_parrafo = False
                self.buffer: list[str] = []

            def handle_starttag(self, tag: str, attrs: Any) -> None:
                if tag in {"script", "style", "nav", "aside", "header", "footer", "form"}:
                    self.ignorar += 1
                elif tag == "p" and not self.ignorar:
                    self.en_parrafo = True
                elif tag == "title":
                    self.en_parrafo = True
                    recolector.capturando_titulo = True

            def handle_endtag(self, tag: str) -> None:
                if tag in {"script", "style", "nav", "aside", "header", "footer", "form"}:
                    self.ignorar = max(0, self.ignorar - 1)
                elif tag == "p" and self.en_parrafo:
                    self.en_parrafo = False
                    texto = "".join(self.buffer).strip()
                    self.buffer.clear()
                    if len(texto) > 40:
                        recolector.parrafos.append(texto)
                elif tag == "title":
                    self.en_parrafo = False
                    recolector.titulo = "".join(self.buffer).strip()
                    self.buffer.clear()
                    recolector.capturando_titulo = False

            def handle_data(self, data: str) -> None:
                if self.en_parrafo and not self.ignorar:
                    self.buffer.append(data)

        self.parrafos: list[str] = []
        self.titulo: str = ""
        self.capturando_titulo = False
        self.parser = _Parser()

    def alimentar(self, html: str) -> None:
        try:
            self.parser.feed(html)
        except Exception:
            pass


def _extraer_con_respaldo(html: str, articulo: ArticuloExtraido) -> None:
    recolector = _RecolectorTexto()
    recolector.alimentar(html)
    articulo.texto = "\n\n".join(recolector.parrafos)
    articulo.titulo = recolector.titulo


# ---------------------------------------------------------------------------
# Limpieza y diagnóstico
# ---------------------------------------------------------------------------


def limpiar_boilerplate(texto: str) -> str:
    """Elimina bloques residuales típicos de medios chilenos."""
    if not texto:
        return ""

    limpio = texto
    for patron in _PATRONES_BOILERPLATE:
        limpio = patron.sub("", limpio)

    lineas: list[str] = []
    for linea in limpio.split("\n"):
        despojada = linea.strip()
        if not despojada:
            lineas.append("")
            continue
        # Una línea corta en mayúsculas sin punto final es casi siempre una
        # etiqueta de sección ("POLÍTICA", "NACIONAL"), no prosa del artículo.
        if len(despojada) < 32 and despojada.isupper() and not despojada.endswith("."):
            continue
        lineas.append(despojada)

    limpio = "\n".join(lineas)
    limpio = re.sub(r"\n{3,}", "\n\n", limpio)
    return limpio.strip()


def _diagnosticar(html: str, articulo: ArticuloExtraido) -> str:
    n = articulo.n_caracteres

    if n == 0:
        articulo.advertencias.append(
            "No se recuperó cuerpo del artículo. Puede tratarse de una página "
            "renderizada por JavaScript o de un formato no soportado."
        )
        return "VACIO"

    plegado = _plegar(f"{html[:200000]} {articulo.texto}")
    tiene_marcador_pago = any(m in plegado for m in _MARCADORES_PAYWALL)

    if tiene_marcador_pago and n < UMBRAL_TRUNCADO:
        articulo.advertencias.append(
            "Se detectó muro de pago y el cuerpo recuperado es breve. Las "
            "entidades corresponden solo al extracto visible."
        )
        return "PAYWALL"

    if n < MINIMO_CARACTERES:
        articulo.advertencias.append(
            f"El cuerpo recuperado tiene {n} caracteres, bajo el mínimo de "
            f"{MINIMO_CARACTERES}. No es analizable con confianza."
        )
        return "VACIO"

    if n < UMBRAL_TRUNCADO:
        articulo.advertencias.append(
            f"El cuerpo recuperado tiene {n} caracteres. Probablemente sea un "
            "extracto; la cobertura de entidades será parcial."
        )
        return "TRUNCADO"

    return "COMPLETO"


def _texto(valor: Any) -> str:
    if valor is None:
        return ""
    return re.sub(r"\s+", " ", str(valor)).strip()


def _plegar(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", str(texto or ""))
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn").casefold()


# ---------------------------------------------------------------------------
# Entrada pública
# ---------------------------------------------------------------------------


def extraer_articulo(url: str, timeout: int = 20) -> ArticuloExtraido:
    """Descarga y extrae un artículo. Nunca lanza excepción."""
    html, error = descargar_html(url, timeout=timeout)
    if error:
        articulo = ArticuloExtraido(url=url, estado_extraccion="ERROR")
        articulo.advertencias.append(error)
        return articulo
    return extraer_desde_html(html, url=url)


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    if len(sys.argv) < 2:
        print("Uso: python extractor_articulo.py <url>")
        raise SystemExit(2)
    resultado = extraer_articulo(sys.argv[1])
    print(json.dumps(resultado.a_dict(), ensure_ascii=False, indent=2))
