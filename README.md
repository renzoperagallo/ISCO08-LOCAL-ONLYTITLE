# ISCO08 LOCAL ONLY TITLE

Clasificador local de títulos de empleo en códigos **CIUO-08.CL**, usando
solamente el título del anuncio.

El flujo es:

```text
title → embedding → distancia coseno → top-10 → Gemma → código
```

No usa `body`, no extrae tareas, no detecta jefaturas y no hace una segunda
reclasificación. Cada título genera **una sola llamada a Gemma**.

## Componentes

- **Embeddings:** `ibm-granite/granite-embedding-311m-multilingual-r2`
  (768 dimensiones, CPU).
- **Retriever:** distancia coseno contra los 444 códigos CIUO-08.CL.
- **Generador:** `hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S` en Ollama, con
  razonamiento (`<|think|>`).
- **Taxonomía:** 444 grupos primarios, 2.325 ejemplos de cargos.

## Flujo

1. Leer `id,title`.
2. Embeber el título.
3. Calcular la distancia coseno contra los 444 códigos.
4. Ordenar de menor a mayor distancia y recuperar top-20.
5. Enviar los top-10 a Gemma.
6. Parsear el código de 4 dígitos.
7. Validar y reparar por prefijo si es necesario.
8. Guardar checkpoint, CSV, XLSX y resumen.

La distancia usada es:

```text
d = 1 - similitud_coseno
```

Los embeddings están normalizados L2, así que:

```text
d = 1 - (vector_título · vector_código)
```

## Entrada

```csv
id,title
1,Recepcionista
2,Jefe de mantenimiento
3,Operario de línea de envasado
```

Archivo incluido:

```text
data/input/sample10.csv
```

El archivo completo de 1.000 títulos **no se incluye** en el repositorio. Para
usarlo, pasa tu propio CSV con `INPUT=...`.

## Instalación

```bash
make install
```

Se reutiliza el servidor Ollama existente en `127.0.0.1:11434`. El modelo debe
estar descargado:

```bash
ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S
```

Comprobar:

```bash
make ollama-check
```

## Uso

```bash
make smoke     # primeros 10 títulos incluidos
make run INPUT=ruta/a/tu_archivo.csv RUN_NAME=mi_corrida
make resume INPUT=ruta/a/tu_archivo.csv RUN_NAME=mi_corrida
```

También se puede cambiar el tamaño de lote:

```bash
make run INPUT=ruta/a/tu_archivo.csv BATCH_SIZE=10 RUN_NAME=mi_corrida
```

## Prompt de clasificación

En modo razonador, Ollama antepone `<|think|>` al system message:

```text
<|think|>
You classify job titles into CIUO-08.CL occupation codes. Think briefly: at most 2-3 short sentences, then return only the 4-digit code.
```

User message:

```text
You are an expert assigning CIUO-08.CL labels (the Chilean adaptation of ISCO-08) to job titles.
The job title may be in Spanish or Portuguese. Select the single best 4-digit CIUO-08.CL code.

JOB TITLE:
{title}

CANDIDATES:
{candidate_block}

Rules:
- Use the candidate descriptions and example jobs to choose the best match.
- The job title may be short, generic or contain spelling errors.
- Pick one candidate even if you are not fully sure.
- Return ONLY the 4-digit code. Do not write explanations, labels or punctuation.
```

Cada candidato se formatea así:

```text
Code: 4226
Label: Recepcionistas (funciones generales)
Description: ...
Example jobs: ...
```

Se incluyen 10 candidatos, descripciones de hasta 900 caracteres y hasta 10
ejemplos por código.

## Salida

```text
outputs/predictions_titles.csv
outputs/predictions_titles.xlsx
outputs/predictions.csv          # copia estable
outputs/predictions.xlsx
outputs/checkpoints/titles.jsonl
outputs/run_summary_titles.json
```

Columnas principales:

```text
id, title, code, label,
distance, similarity,
corrected, fallback_used,
llm_response, top_candidates,
prompt_tokens, completion_tokens,
latency_s, prompt_eval_s, eval_s,
generation_tokens_per_s
```

- `distance`: distancia coseno del top-1.
- `similarity`: `1 - distance`.
- `corrected`: el código se reparó por prefijo.
- `fallback_used`: no se pudo parsear/reparar y se usó el top-1.

## Checkpoint

`make run INPUT=...` empieza de cero: borra checkpoint y salidas.
`make resume INPUT=...` continúa desde el checkpoint de ese `RUN_NAME`.

El runner procesa por lotes. Si se interrumpe durante las llamadas de un lote,
ese lote puede repetirse; los lotes ya cerrados no se recalculan.

## Resultado de la prueba de 10 títulos

La prueba `make smoke` se ejecutó correctamente:

```text
Títulos procesados:       10
Códigos válidos:          10
Corregidos por prefijo:    0
Fallbacks:                 0
Distancia media:           0.136554
Similitud media:           0.863446
Latencia media:            11.66 s
Latencia p95:              18.43 s
Tiempo total:              126.37 s
```

Ejemplos:

| Título | Código | Etiqueta | Distancia |
|---|---|---:|---:|
| Recepcionista | 4226 | Recepcionistas (funciones generales) | 0.071590 |
| Enfermera profesional pediátrica | 2221 | Enfermeros profesionales | 0.134047 |
| Soldador de caños con Mig Mag | 7212 | Soldadores y oxicortadores | 0.152206 |
| Representante de Venta Telefónica | 5244 | Vendedores por internet y otros medios de comunicación | 0.112650 |
| Mecánico Oficial de Primera | 7231 | Mecánicos y reparadores de vehículos de motor | 0.145244 |

La salida de esta prueba está en:

```text
outputs/predictions_smoke10.csv
outputs/predictions_smoke10.xlsx
outputs/run_summary_smoke10.json
```

## Tests

```bash
make test
```

## Limitaciones

- Solo se usa el título; títulos genéricos como "Operario" o "Asistente"
  pueden ser ambiguos.
- CIUO-08.CL es la adaptación chilena; anuncios de otros países pueden no
  encajar perfectamente.
- La `distance` es una distancia de embeddings, no una probabilidad.
- El razonamiento local agrega latencia.
- Una corrida grande puede tardar horas en una GPU de 6 GB.

## Licencia

AGPL-3.0. Ver `LICENSE`.
