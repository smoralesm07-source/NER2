#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Medición contra gold standard anotado.

Por qué existe este archivo
---------------------------
Sin esto no puedes afirmar "alta precisión", y la suite de regresión no
sustituye la medición: un F1 de 1.000 sobre el corpus de pruebas significa
"no hay regresiones", no "funciona en prensa real". Son cosas distintas y
confundirlas es la forma más común de creer que un sistema de extracción anda
bien cuando no anda bien.

Flujo de trabajo
----------------
1.  Crear la plantilla de anotación a partir de artículos reales::

        python evaluar.py plantilla --entradas resultados/*.json --salida gold/

2.  Anotar a mano cada archivo de ``gold/``. Un anotador marca todas las
    entidades del artículo con su tipo. Anota lo que el artículo dice, no lo
    que el sistema propuso: los campos vienen prellenados solo como punto de
    partida y hay que corregirlos, incluido borrar los falsos positivos y
    agregar lo que el sistema no vio.

3.  Medir::

        python evaluar.py medir --gold gold/ --predicho resultados/

Criterio de coincidencia
------------------------
Se reportan dos, porque miden cosas distintas y elegir solo uno oculta
información:

``estricto``
    Los offsets deben coincidir exactamente. Es el criterio duro.

``relajado``
    Basta que los spans se solapen. Refleja utilidad operativa: si el sistema
    devolvió "Ricardo Bustamante" donde el gold dice "Ricardo Bustamante
    Leiva", el analista igual encuentra a la persona.

Tamaño mínimo
-------------
Con menos de ~120 artículos los intervalos de confianza son tan anchos que las
comparaciones entre versiones no significan nada. Apunta a 150–200, y haz que
al menos el 20% lo anoten dos personas para poder medir el acuerdo entre
anotadores: si dos analistas no coinciden entre sí, el techo del sistema es ese
desacuerdo, no el 100%.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any

VERSION_EVALUADOR = "1.0.0"


# ---------------------------------------------------------------------------
# Plantilla
# ---------------------------------------------------------------------------


def crear_plantilla(rutas: list[str], destino: str) -> int:
    os.makedirs(destino, exist_ok=True)
    creados = 0

    for ruta in rutas:
        with open(ruta, encoding="utf-8") as fh:
            resultado = json.load(fh)

        nombre = os.path.splitext(os.path.basename(ruta))[0] + ".gold.json"
        salida = os.path.join(destino, nombre)
        if os.path.exists(salida):
            print(f"  omitido (ya existe): {salida}")
            continue

        plantilla = {
            "_instrucciones": [
                "Anota TODAS las entidades del artículo, no solo las que el sistema propuso.",
                "Borra las entradas que sean falsos positivos.",
                "Agrega las que falten, con inicio/fin exactos sobre 'texto_analizado'.",
                "Corrige el tipo cuando corresponda. El tipo es lo que se mide.",
                "Deja 'anotador' con tu nombre y marca 'revisado': true al terminar.",
            ],
            "articulo_url": resultado.get("articulo", {}).get("url", ""),
            "anotador": "",
            "revisado": False,
            "texto_analizado": resultado.get("texto_analizado", ""),
            "entidades": [
                {
                    "texto": e["texto"],
                    "inicio": e["inicio"],
                    "fin": e["fin"],
                    "tipo": e["tipo"],
                    "naturaleza": e["naturaleza"],
                    "rol_procesal": e.get("rol_procesal", "SIN_ROL"),
                    "_propuesto_por_el_sistema": True,
                }
                for e in resultado.get("entidades", [])
            ],
        }

        with open(salida, "w", encoding="utf-8") as fh:
            json.dump(plantilla, fh, ensure_ascii=False, indent=2)
        creados += 1
        print(f"  creado: {salida}")

    return creados


# ---------------------------------------------------------------------------
# Emparejamiento
# ---------------------------------------------------------------------------


def _solapan(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (a["fin"] <= b["inicio"] or a["inicio"] >= b["fin"])


def _exactos(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["inicio"] == b["inicio"] and a["fin"] == b["fin"]


def emparejar(
    gold: list[dict[str, Any]], predicho: list[dict[str, Any]], estricto: bool
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Devuelve ``(pares, no_encontradas, sobrantes)``.

    El emparejamiento es 1:1 y voraz sobre el orden del gold. Una predicción ya
    emparejada no se reutiliza, para que duplicar una entidad cuente como falso
    positivo y no como acierto extra.
    """
    coincide = _exactos if estricto else _solapan
    disponibles = list(predicho)
    pares: list[tuple[dict, dict]] = []
    no_encontradas: list[dict] = []

    for referencia in gold:
        candidatos = [p for p in disponibles if coincide(referencia, p)]
        if not candidatos:
            no_encontradas.append(referencia)
            continue
        # Ante varios, el de mayor solapamiento.
        mejor = max(
            candidatos,
            key=lambda p: min(referencia["fin"], p["fin"]) - max(referencia["inicio"], p["inicio"]),
        )
        disponibles.remove(mejor)
        pares.append((referencia, mejor))

    return pares, no_encontradas, disponibles


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def _prf(vp: int, fp: int, fn: int) -> dict[str, float]:
    precision = vp / (vp + fp) if (vp + fp) else 0.0
    recall = vp / (vp + fn) if (vp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def medir(gold_dir: str, predicho_dir: str) -> dict[str, Any]:
    archivos_gold = sorted(glob.glob(os.path.join(gold_dir, "*.gold.json")))
    if not archivos_gold:
        raise SystemExit(f"No se encontraron archivos *.gold.json en {gold_dir}")

    informe: dict[str, Any] = {}

    for modo, estricto in (("estricto", True), ("relajado", False)):
        vp = fp = fn = 0
        por_tipo: dict[str, dict[str, int]] = defaultdict(lambda: {"vp": 0, "fp": 0, "fn": 0})
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        sin_revisar: list[str] = []
        errores: list[dict[str, Any]] = []

        for ruta_gold in archivos_gold:
            with open(ruta_gold, encoding="utf-8") as fh:
                referencia = json.load(fh)

            if not referencia.get("revisado"):
                sin_revisar.append(os.path.basename(ruta_gold))
                continue

            base = os.path.basename(ruta_gold).replace(".gold.json", ".json")
            ruta_pred = os.path.join(predicho_dir, base)
            if not os.path.exists(ruta_pred):
                print(f"  aviso: falta la predicción {ruta_pred}")
                continue

            with open(ruta_pred, encoding="utf-8") as fh:
                prediccion = json.load(fh)

            pares, faltantes, sobrantes = emparejar(
                referencia["entidades"], prediccion.get("entidades", []), estricto
            )

            for esperada, obtenida in pares:
                tipo_esperado = esperada["tipo"]
                tipo_obtenido = obtenida["tipo"]
                confusion[esperada.get("naturaleza", "?")][obtenida.get("naturaleza", "?")] += 1

                if tipo_esperado == tipo_obtenido:
                    vp += 1
                    por_tipo[tipo_esperado]["vp"] += 1
                else:
                    # Span correcto pero tipo equivocado: cuenta doble, como
                    # falso negativo del tipo real y falso positivo del asignado.
                    fp += 1
                    fn += 1
                    por_tipo[tipo_esperado]["fn"] += 1
                    por_tipo[tipo_obtenido]["fp"] += 1
                    errores.append(
                        {
                            "archivo": base,
                            "texto": esperada["texto"],
                            "esperado": tipo_esperado,
                            "obtenido": tipo_obtenido,
                        }
                    )

            for esperada in faltantes:
                fn += 1
                por_tipo[esperada["tipo"]]["fn"] += 1
                errores.append(
                    {
                        "archivo": base,
                        "texto": esperada["texto"],
                        "esperado": esperada["tipo"],
                        "obtenido": "NO_DETECTADA",
                    }
                )

            for sobrante in sobrantes:
                fp += 1
                por_tipo[sobrante["tipo"]]["fp"] += 1
                errores.append(
                    {
                        "archivo": base,
                        "texto": sobrante["texto"],
                        "esperado": "NO_ES_ENTIDAD",
                        "obtenido": sobrante["tipo"],
                    }
                )

        informe[modo] = {
            "global": {**_prf(vp, fp, fn), "vp": vp, "fp": fp, "fn": fn},
            "por_tipo": {
                tipo: {**_prf(c["vp"], c["fp"], c["fn"]), **c}
                for tipo, c in sorted(por_tipo.items())
            },
            "matriz_naturaleza": {k: dict(v) for k, v in confusion.items()},
            "archivos_sin_revisar": sin_revisar,
            "errores": errores[:80],
        }

    informe["n_articulos"] = len(archivos_gold)
    return informe


def imprimir(informe: dict[str, Any]) -> None:
    print(f"\nArtículos en el gold standard: {informe['n_articulos']}")
    if informe["n_articulos"] < 120:
        print(
            "  ! Bajo 120 artículos los intervalos de confianza son demasiado\n"
            "    anchos para comparar versiones con sentido."
        )

    for modo in ("estricto", "relajado"):
        datos = informe[modo]
        g = datos["global"]
        print(f"\n{'=' * 68}\nCriterio {modo.upper()}")
        print(
            f"  precision {g['precision']:.3f}   recall {g['recall']:.3f}   "
            f"F1 {g['f1']:.3f}   (VP {g['vp']}  FP {g['fp']}  FN {g['fn']})"
        )

        if datos["archivos_sin_revisar"]:
            print(
                f"  ! {len(datos['archivos_sin_revisar'])} archivo(s) sin marcar "
                "'revisado': true, excluidos del cálculo."
            )

        print(f"\n  {'TIPO':<30}{'PREC':>7}{'REC':>8}{'F1':>8}{'VP':>6}{'FP':>5}{'FN':>5}")
        for tipo, m in datos["por_tipo"].items():
            print(
                f"  {tipo:<30}{m['precision']:>7.3f}{m['recall']:>8.3f}"
                f"{m['f1']:>8.3f}{m['vp']:>6}{m['fp']:>5}{m['fn']:>5}"
            )

        # El número que responde la pregunta original.
        print("\n  Matriz de confusión por naturaleza (fila = real, columna = predicha)")
        matriz = datos["matriz_naturaleza"]
        columnas = sorted({c for fila in matriz.values() for c in fila})
        if columnas:
            print(f"  {'':<20}" + "".join(f"{c[:16]:>18}" for c in columnas))
            for real in sorted(matriz):
                print(
                    f"  {real[:19]:<20}"
                    + "".join(f"{matriz[real].get(c, 0):>18}" for c in columnas)
                )

    print("\nPrimeros errores (criterio relajado):")
    for err in informe["relajado"]["errores"][:20]:
        print(f"  «{err['texto'][:38]:<40}» esperado {err['esperado']:<26} → {err['obtenido']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mide el pipeline contra un gold standard.")
    sub = parser.add_subparsers(dest="comando", required=True)

    p1 = sub.add_parser("plantilla", help="Crear plantillas de anotación")
    p1.add_argument("--entradas", nargs="+", required=True)
    p1.add_argument("--salida", default="gold")

    p2 = sub.add_parser("medir", help="Calcular métricas")
    p2.add_argument("--gold", default="gold")
    p2.add_argument("--predicho", default="resultados")
    p2.add_argument("--informe", help="Ruta para guardar el informe en JSON")

    args = parser.parse_args(argv)

    if args.comando == "plantilla":
        rutas: list[str] = []
        for patron in args.entradas:
            rutas.extend(glob.glob(patron))
        print(f"Creando plantillas en {args.salida}/")
        print(f"Listo: {crear_plantilla(rutas, args.salida)} plantilla(s).")
        return 0

    informe = medir(args.gold, args.predicho)
    imprimir(informe)
    if args.informe:
        with open(args.informe, "w", encoding="utf-8") as fh:
            json.dump(informe, fh, ensure_ascii=False, indent=2)
        print(f"\nInforme guardado en {args.informe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
