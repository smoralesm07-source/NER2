#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Servidor local del prototipo — solo biblioteca estándar.

Por qué no FastAPI
------------------
Para un endpoint único en un servidor institucional, FastAPI arrastra uvicorn,
starlette y pydantic sin aportar nada que ``http.server`` no resuelva. Menos
dependencias significa menos superficie que auditar y un despliegue que
funciona en cualquier máquina con Python 3.10+, sin instalar nada.

Uso
---
    export ANTHROPIC_API_KEY="sk-ant-..."
    python servidor_local.py --puerto 8000

Luego abrir http://127.0.0.1:8000

Enlace público
--------------
El servidor escucha en 127.0.0.1 de forma deliberada. No expone autenticación
ni TLS; publicarlo en una interfaz de red accesible dejaría la credencial de la
API y el índice de entidades al alcance de cualquiera en el segmento. Para uso
compartido en la institución, ponerlo detrás de un proxy inverso con
autenticación.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pipeline_url

RAIZ = Path(__file__).resolve().parent
ARCHIVO_INTERFAZ = RAIZ / "analizar_url.html"

LIMITE_CUERPO = 2 * 1024 * 1024  # 2 MB


class Manejador(BaseHTTPRequestHandler):
    server_version = "MonitorUAF-Prototipo/1.0"

    # -- utilidades ---------------------------------------------------------

    def _responder(self, codigo: int, cuerpo: bytes, tipo: str) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, codigo: int, datos: dict) -> None:
        self._responder(
            codigo,
            json.dumps(datos, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def log_message(self, formato: str, *args) -> None:
        # La URL analizada puede contener el nombre de un investigado. No se
        # escribe en el log de acceso.
        sys.stderr.write(f"{self.address_string()} - {args[0].split(' ')[0]} {args[1]}\n")

    # -- rutas --------------------------------------------------------------

    def do_GET(self) -> None:
        ruta = self.path.split("?")[0]

        if ruta in ("/", "/index.html", "/analizar_url.html"):
            if not ARCHIVO_INTERFAZ.exists():
                self._responder(
                    404,
                    b"No se encuentra analizar_url.html junto a servidor_local.py.",
                    "text/plain; charset=utf-8",
                )
                return
            self._responder(
                200,
                ARCHIVO_INTERFAZ.read_bytes(),
                "text/html; charset=utf-8",
            )
            return

        if ruta == "/estado":
            self._json(
                200,
                {
                    "version_pipeline": pipeline_url.VERSION_PIPELINE,
                    "gliner_disponible": pipeline_url.capa_gliner.disponible(),
                    "gliner_error": pipeline_url.capa_gliner.error_carga(),
                    "api_key_presente": bool(os.environ.get("ANTHROPIC_API_KEY")),
                    "reglas": pipeline_url.capa_reglas.diagnostico(),
                },
            )
            return

        self._json(404, {"error": "Ruta no encontrada."})

    def do_POST(self) -> None:
        if self.path.split("?")[0] != "/analizar":
            self._json(404, {"error": "Ruta no encontrada."})
            return

        try:
            largo = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            largo = 0

        if largo <= 0 or largo > LIMITE_CUERPO:
            self._json(400, {"error": "Cuerpo de la petición ausente o demasiado grande."})
            return

        try:
            peticion = json.loads(self.rfile.read(largo).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "El cuerpo debe ser JSON válido."})
            return

        url = str(peticion.get("url", "")).strip()
        texto = str(peticion.get("texto", "")).strip()

        if not url and not texto:
            self._json(400, {"error": "Indica una URL o pega el texto del artículo."})
            return

        opciones = {
            "usar_gliner": bool(peticion.get("usar_gliner", True)),
            "usar_llm": bool(peticion.get("usar_llm", True)),
            "umbral_gliner": float(
                peticion.get("umbral_gliner", pipeline_url.capa_gliner.UMBRAL_POR_DEFECTO)
            ),
        }

        try:
            resultado = (
                pipeline_url.analizar_url(url, **opciones)
                if url
                else pipeline_url.analizar_texto(texto, **opciones)
            )
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            self._json(500, {"error": f"El análisis falló: {exc}"})
            return

        self._json(200, resultado)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Servidor del prototipo de análisis de prensa.")
    parser.add_argument("--puerto", type=int, default=8000)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Cambiar solo detrás de un proxy con autenticación.",
    )
    parser.add_argument(
        "--precargar-gliner",
        action="store_true",
        help="Cargar el modelo al iniciar en vez de en la primera consulta.",
    )
    args = parser.parse_args(argv)

    if args.precargar_gliner:
        print("Cargando GLiNER...", file=sys.stderr)
        disponible = pipeline_url.capa_gliner.disponible()
        print(
            "GLiNER listo." if disponible
            else f"GLiNER no disponible: {pipeline_url.capa_gliner.error_carga()}",
            file=sys.stderr,
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Aviso: ANTHROPIC_API_KEY no está definida. La capa de adjudicación "
            "no se ejecutará y el análisis quedará limitado a reglas y GLiNER.",
            file=sys.stderr,
        )

    servidor = ThreadingHTTPServer((args.host, args.puerto), Manejador)
    print(f"Escuchando en http://{args.host}:{args.puerto}", file=sys.stderr)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nDetenido.", file=sys.stderr)
    finally:
        servidor.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
