# Reconocimiento de entidades en prensa · Prototipo v1.1.0

Recibe el enlace de una noticia y devuelve las entidades nombradas
clasificadas, distinguiendo **personas naturales de personas jurídicas**, con
su rol procesal y las relaciones explícitas entre ellas.

Construido para acoplarse al Monitor UAF Chile, con degradación controlada en
cada capa: funciona con lo que esté instalado y declara en la salida qué corrió
y qué no.

---

## 1. Instalación sin entorno local

Todo el flujo asume que no tienes Python instalado en tu máquina y que trabajas
por la interfaz web de GitHub.

**Subir los archivos**

1. En el repositorio, `Add file` → `Upload files`.
2. Arrastra los `.py`, el `.html` y `requirements_pipeline.txt` a la raíz.
3. `.github/workflows/analizar_noticia.yml` va en esa ruta exacta. GitHub crea
   las carpetas si escribes la ruta completa en el nombre del archivo.
4. `fixtures/nota_ejemplo.html` va en `fixtures/`.

**Configurar el secreto**

`Settings` → `Secrets and variables` → `Actions` → `New repository secret`.
Nombre: `ANTHROPIC_API_KEY`. Sin este secreto el pipeline corre igual, pero sin
la capa de adjudicación.

**Analizar una noticia**

`Actions` → `Analizar noticia (entidades)` → `Run workflow` → pegar la URL.
El resultado queda como artefacto descargable (`resultado.json`).

**Ver el resultado**

Abre `analizar_url.html` en el navegador y usa el botón **Abrir JSON**. No
necesita servidor para esto.

**Ver la interfaz de inmediato**

`demo_resultado.json` viene incluido con un caso completo ya procesado. Ábrelo
con ese mismo botón para revisar la interfaz antes de configurar nada.

---

## 2. Arquitectura

```
  URL
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│ L0  extractor_articulo.py                                        │
│     trafilatura · limpieza de boilerplate chileno                │
│     diagnóstico: COMPLETO / TRUNCADO / PAYWALL / VACIO           │
└──────────────────────────────────────────────────────────────────┘
   │  texto limpio
   ├──────────────────────┬───────────────────────┐
   ▼                      ▼                       │
┌──────────────────┐  ┌──────────────────┐        │
│ L1 capa_reglas   │  │ L2 capa_gliner   │        │
│ sufijos, RUT,    │  │ zero-shot, 11    │        │
│ encabezados      │  │ etiquetas ES     │        │
│ precisión ~99%   │  │ recall alto      │        │
│ recall bajo      │  │                  │        │
└──────────────────┘  └──────────────────┘        │
   │  candidatos          │  candidatos           │
   └──────────┬───────────┘                       │
              ▼                                   │
┌──────────────────────────────────────────────────────────────────┐
│ L3  adjudicador_llm.py    ◄─── texto completo ───────────────────┘
│     naturaleza jurídica · correferencia · ROL PROCESAL           │
│     relaciones · structured outputs con JSON Schema              │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ L3b validador_spans.py   BARRERA                                 │
│     texto[inicio:inicio+len(span)] == span  →  si no, se descarta│
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ L4  fusion_entidades.py                                          │
│     precedencia REGLA > LLM > GLINER · confianza · cola revisión │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
        entidades clasificadas
```

### Las dos garantías que sostienen el diseño

**Anclaje verificable.** Toda entidad debe traer su cadena literal y su offset,
y se comprueba en Python con igualdad de strings que esté en el artículo. Lo
que no se verifica, no se emite. El modelo no puede eludir esta comprobación
argumentando: no es un juicio, es `==`. La misma comprobación se aplica a la
frase de evidencia y a las relaciones, porque anclar bien el nombre e inventar
la oración que sustenta el rol es el error con consecuencias.

**Precedencia determinista.** Si la capa de reglas afirma que
`Inversiones Muñoz Ltda.` es persona jurídica por su sufijo societario, esa
clasificación se mantiene aunque el adjudicador argumente lo contrario. El
sufijo es un hecho del texto; la lectura del modelo es una inferencia. El
conflicto no se oculta: queda en `conflictos` y marca `requiere_validacion`.

---

## 3. Elección del motor L2: GLiNER2 o GLiNER v1

El pipeline soporta ambos con la misma interfaz. Se elige con
`GLINER_BACKEND=v2` (por defecto) o `GLINER_BACKEND=v1`. Si el elegido no está
instalado se prueba el otro antes de caer a spaCy.

### GLiNER2 (`capa_gliner2.py`) — recomendado

Ventaja concreta sobre v1: **cada etiqueta admite una descripción**. En vez de
la cadena suelta `"persona"` se pasa una definición que enuncia el caso
difícil. `ESQUEMA_UAF` lo usa para decir explícitamente que una razón social con
apellidos es empresa y que las comunas con nombre de personaje histórico son
lugares. Con v1 eso solo se puede esperar de la inferencia del modelo.

**Tres trampas verificadas en el código de `gliner2` 1.3.2:**

| Trampa | Consecuencia si se ignora |
|---|---|
| El modelo del README (`fastino/gliner2-base-v1`) está etiquetado **English** en Hugging Face | Rinde mal sobre prensa chilena y se concluye, equivocadamente, que GLiNER2 no sirve. Hay que usar `fastino/gliner2-multi-v1`, que no aparece en el README |
| `include_spans` es `False` por defecto | La salida no trae offsets de carácter. Sin ellos, la barrera de anclaje y la fusión por solapamiento no pueden operar: el pipeline completo deja de funcionar |
| La salida viene agrupada por etiqueta, no como lista plana | `{"entities": {"persona": [...], "empresa": [...]}}` en vez de `[{start, end, label}, ...]`. Hay que aplanarla |

Las tres están resueltas en `capa_gliner2.py`, y hay pruebas de regresión que
impiden que reaparezcan: una verifica que `include_spans=True` se pase siempre,
otra que el modelo por defecto sea el multilingüe.

GLiNER2 también trae clasificación de texto y extracción de relaciones en la
misma pasada. **No se cablean a propósito:** sus relaciones no vienen con la
frase del artículo que las sustenta, y sin evidencia verificable una relación
no es utilizable para análisis AML. Las relaciones siguen saliendo de L3, donde
cada una se descarta si su frase de sustento no existe en el texto.

### GLiNER v1 (`capa_gliner.py`)

Sigue disponible con `GLINER_BACKEND=v1` y `urchade/gliner_multi-v2.1`. Su
límite duro es `max_len = 384` tokens, verificado en el código de `gliner`
0.2.28; ver la nota sobre chunking más abajo.

---

## 3b. Por qué cualquiera de los dos reemplaza a `es_core_news_sm`

El modelo pequeño de spaCy solo conoce cuatro clases (PER, ORG, LOC, MISC).
Para producir `ORGANISMO_PUBLICO` o `INSTITUCION_FINANCIERA` hay que
post-procesar ORG con reglas, y ahí se acumulan los errores. Su confusión
PER↔ORG está documentada justo en el caso que más importa: razones sociales que
contienen antropónimos.

GLiNER es zero-shot: las etiquetas se definen en tiempo de inferencia. Se le
entrega la taxonomía UAF en español y clasifica contra ella sin reentrenar y
sin corpus anotado.

### Chunking obligatorio, no opcional

Verificado leyendo el código de `gliner` 0.2.28: `max_len` por defecto es **384
tokens**. Un artículo de prensa chileno promedia entre 500 y 1.500. Pasarle el
texto completo **trunca en silencio** todo lo que exceda el límite: se pierden
entidades sin ningún error visible.

`capa_gliner.py` segmenta en 1.100 caracteres con 180 de solape, respetando
frontera de oración, y **remapea los offsets al texto original**. Hay una
prueba que verifica `texto[offset:offset+len(fragmento)] == fragmento` para cada
segmento; si eso se rompiera, el validador de spans rechazaría entidades
legítimas y el pipeline perdería recall sin dar señal.

### Costo

`torch` son ~200 MB y los pesos ~500 MB, descargados de Hugging Face en la
primera ejecución. En GitHub Actions hay que cachear `~/.cache/huggingface` o
cada corrida vuelve a bajarlos; el workflow ya lo hace.

Si no puedes instalar torch, el respaldo con spaCy funciona pero rinde bastante
menos, y mapea deliberadamente `ORG → OTRO` en vez de `ORG → EMPRESA`: hacer
que ORG caiga en EMPRESA introduce un sesgo sistemático hacia empresa privada
que castiga a organismos públicos y tribunales.

---

## 4. Acoplamiento con el Monitor UAF

Los tipos canónicos son un **espejo exacto** de `NATURALEZA_POR_TIPO` en
`reconocedor_entidades.py` v3. `entidades.html` y el consumo actual de
`datos.json` siguen funcionando sin cambios: los campos `texto`, `tipo`,
`naturaleza`, `confianza_score`, `senales`, `ruts` y `requiere_validacion`
conservan su forma.

Campos nuevos disponibles: `rol_procesal`, `evidencia`, `justificacion`,
`fuentes`, `anclaje`, `conflictos`, `motivos_validacion`, `nombre_normalizado`,
`variantes`.

`capa_reglas.py` importa `reconocedor_entidades` y `geografia_cl` si están en
el path, y opera con su propio conjunto reducido si no. `capa_reglas.diagnostico()`
te dice cuál de los dos casos ocurrió.

**Para integrarlo al monitor de prensa**, llama a `pipeline_url.analizar_texto()`
sobre el cuerpo que ya levantas, en vez de a `analizar_url()`. El extractor L0
se salta y todo lo demás corre igual.

---

## 5. Uso local

```bash
pip install -r requirements_pipeline.txt
export ANTHROPIC_API_KEY="sk-ant-..."

# Una noticia
python pipeline_url.py --url https://... --salida resultado.json --resumen

# Sin capas opcionales
python pipeline_url.py --archivo fixtures/nota_ejemplo.html --sin-gliner --sin-llm --resumen

# Servidor + interfaz
python servidor_local.py --puerto 8000 --precargar-gliner
# abrir http://127.0.0.1:8000

# Suite de regresión
python -m unittest test_pipeline_url -v
```

El servidor escucha en `127.0.0.1` a propósito: no tiene autenticación ni TLS.
Para uso compartido en la institución va detrás de un proxy inverso con
autenticación.

---

## 6. Medir de verdad

**La suite de 49 pruebas no mide desempeño de reconocimiento.** Verifica que el
anclaje descarte lo que no está, que la precedencia se respete, que los roles
prohibidos se anulen, que el chunking remapee bien y que cada capa degrade sin
arrastrar a las demás. Pasar las 49 no dice nada sobre precision y recall en
prensa real.

Para poder afirmar "alta precisión" necesitas el gold standard:

```bash
# 1. Analizar 150-200 noticias reales de tu monitor
for url in $(cat urls.txt); do
  python pipeline_url.py --url "$url" --salida "resultados/$(date +%s%N).json"
done

# 2. Crear las plantillas de anotación
python evaluar.py plantilla --entradas "resultados/*.json" --salida gold/

# 3. Anotar a mano cada archivo de gold/ (esto es el trabajo real)

# 4. Medir
python evaluar.py medir --gold gold/ --predicho resultados/ --informe informe.json
```

`evaluar.py` reporta precision, recall y F1 por clase bajo dos criterios
(estricto y relajado) y publica la **matriz de confusión PN↔PJ**, que es el
número que responde la pregunta original.

Tres cosas que determinan si la medición sirve:

- **Tamaño.** Bajo 120 artículos los intervalos de confianza son tan anchos que
  comparar versiones no significa nada.
- **Acuerdo entre anotadores.** Haz que al menos el 20% lo anoten dos personas.
  Si dos analistas no coinciden entre sí, el techo del sistema es ese
  desacuerdo, no el 100%.
- **Anotar lo que dice el artículo, no lo que propuso el sistema.** Las
  plantillas vienen prellenadas con la salida del pipeline como punto de
  partida; hay que borrar falsos positivos y agregar lo que no vio. Si solo
  confirmas lo prellenado, mides tu propia credulidad.

**Criterio operativo sugerido:** prioriza precisión en persona jurídica (esas
se cruzan contra tu base de sujetos obligados y un falso positivo contamina el
cruce) y recall en persona natural (perder una mención es perder señal).

---

## 7. Costo del adjudicador

Un artículo de prensa promedia ~1.500 tokens. Con el prompt de instrucciones y
los candidatos, la petición ronda 3.000 tokens de entrada y 800 de salida.

Dos palancas ya implementadas o disponibles:

- **Prompt caching** (implementado). El bloque de instrucciones es idéntico en
  todos los artículos y lleva `cache_control`. En una corrida de 200 noticias
  solo se paga completo la primera vez.
- **Batch API** (disponible, no implementado). 50% de descuento para la corrida
  programada del monitor. Conviene dejar la llamada síncrona solo para consultas
  ad-hoc desde la interfaz.

El modelo por defecto es `claude-sonnet-4-6`, configurable con la variable de
entorno `CLAUDE_MODELO`. Para corridas masivas, `claude-haiku-4-5` baja el costo
de forma sustantiva; **mide la caída de F1 sobre el gold standard antes de
cambiar**, no lo asumas en ninguna dirección.

---

## 8. Cautelas legales

Estas no son decoración: están implementadas como código.

**Presunción de inocencia.** La taxonomía de roles no admite ninguna etiqueta
que impute responsabilidad penal. `FORMALIZADO`, `IMPUTADO` y `ACUSADO`
describen un estado procesal verificable en la fuente. Si el modelo devuelve
`CULPABLE`, `LAVADOR` o `TESTAFERRO`, el rol **se anula** y la entidad se marca
para revisión. No se suaviza a `INVESTIGADO`: eso seguiría afirmando una
calidad procesal cuya única fuente es un juicio del modelo que ya se descartó
por inadmisible. La evidencia sí se conserva, para que el analista decida.

**Lugares y montos no son sujetos de derecho.** Si una entidad de naturaleza
`NO_APLICA` recibe rol procesal, el rol se anula y queda registrado como
conflicto: es síntoma de que el modelo confundió el tipo.

**Ley 21.719.** Un índice persistente de personas naturales asociadas a
menciones penales en prensa constituye tratamiento de datos relativos a
infracciones. La base de licitud está en la Ley 19.913, pero conviene
documentar finalidad, plazo de conservación y el hecho de que una mención de
prensa no equivale a condena. El workflow limita la retención del artefacto a
7 días por esta razón, y el servidor no escribe la URL analizada en el log de
acceso porque puede contener el nombre de un investigado.

**Datos que salen.** El texto de prensa es público, de modo que enviarlo a la
API no plantea el problema del artículo 13 de la Ley 19.913. Lo que **no** debe
pasar por esta ruta es texto de ROS, carpetas investigativas o cualquier
insumo reservado.

---

## 9. Correcciones surgidas de las pruebas

Se documentan porque el mismo tipo de error puede reaparecer al extender el
sistema:

1. **RUT mal atribuido.** Un RUT de persona natural se asoció a "Unidad de
   Análisis Financiero" solo porque el nombre de su titular no había sido
   detectado. Ahora el tramo de numeración (bajo/sobre 50.000.000) veta la
   asociación incoherente y la ventana es asimétrica, porque en español la
   convención es «Nombre, RUT X». Un RUT sin dueño coherente se reporta
   huérfano en vez de atribuirse mal.
2. **Rol procesal en un lugar.** "San Ramón" salió con calidad procesal. Un
   lugar no es sujeto de derecho.
3. **Giro vs. sufijo.** `Sartor Administradora General de Fondos S.A.` caía en
   `EMPRESA` porque `S.A.` ganaba al giro declarado. El giro es información más
   precisa del mismo texto y ahora prevalece.
4. **Confianza contradictoria.** Una entidad podía salir con confianza 1.00 y
   bandera de revisión simultáneamente. Un rol inadmisible ahora castiga la
   confianza: si el modelo erró al punto de proponer un rol inválido, su juicio
   sobre esa entidad merece menos crédito, no el máximo.

---

## 10. Archivos

| Archivo | Capa | Depende de |
|---|---|---|
| `taxonomia_uaf.py` | — | nada |
| `extractor_articulo.py` | L0 | trafilatura (opcional) |
| `capa_reglas.py` | L1 | nada |
| `capa_gliner.py` | L2 | gliner + torch (opcional) |
| `adjudicador_llm.py` | L3 | nada (urllib) |
| `validador_spans.py` | L3b | nada |
| `fusion_entidades.py` | L4 | nada |
| `pipeline_url.py` | orquestador | las anteriores |
| `servidor_local.py` | interfaz | nada |
| `analizar_url.html` | interfaz | nada |
| `evaluar.py` | medición | nada |
| `generar_demo.py` | demostración | nada |
| `test_pipeline_url.py` | 49 pruebas | nada |
