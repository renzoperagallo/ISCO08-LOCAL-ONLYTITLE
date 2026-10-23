# ISCO08 LOCAL ONLY TITLE

Clasificador local de títulos de empleo en códigos **CIUO-08.CL**, usando
solamente el título del anuncio.

```text
title → embedding → distancia coseno → top-10 → Gemma → código
```

No usa `body`, no extrae tareas, no detecta jefaturas y no hace una segunda
reclasificación. Cada título genera **una sola llamada a Gemma**.

## Instalación y configuración

### Requisitos

- Python 3.11.
- Ollama con el modelo `hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S`.
- Opcional: GPU NVIDIA con CUDA para Gemma y, si hay VRAM suficiente, para el
  modelo de embeddings.

### 1. Instalar el proyecto

```bash
make install
```

Esto crea `.venv` e instala las dependencias. Si no existe `.env`, créalo a
partir de la plantilla:

```bash
cp .env.example .env
```

`.env` no se versiona; `.env.example` es la plantilla para otros equipos.

### 2. Descargar el modelo

```bash
ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S
```

Si Ollama está instalado en otra ruta, define `OLLAMA_BIN` y `OLLAMA_MODELS` en
`.env` antes de arrancarlo.

### 3. Configurar el paralelismo

Hay dos niveles:

- `MAX_WORKERS` (cliente): cuántas peticiones `/api/chat` se envían a la vez.
- `OLLAMA_NUM_PARALLEL` (servidor): cuántas peticiones procesa Ollama a la vez.

Para que haya aceleración real:

```text
OLLAMA_NUM_PARALLEL >= MAX_WORKERS
```

El proyecto puede arrancar Ollama con la configuración correcta:

```bash
make ollama-start NUM_PARALLEL=3
# o en segundo plano:
make ollama-start-bg NUM_PARALLEL=3
```

Si `NUM_PARALLEL=0` (el valor por defecto), se usa `OLLAMA_NUM_PARALLEL` del
`.env` y se eleva automáticamente a `MAX_WORKERS` si es menor. El lanzador
también:

- fija `OLLAMA_CONTEXT_LENGTH` desde `OLLAMA_NUM_CTX`;
- usa `OLLAMA_BIN` o `ollama` del `PATH`;
- pasa `OLLAMA_MODELS` si está definido;
- no arranca si el puerto ya está ocupado;
- permite ver el comando con `--dry-run`.

Ejemplo de `.env` para un equipo con VRAM suficiente:

```env
MAX_WORKERS=3
OLLAMA_NUM_PARALLEL=3
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=64
```

Para una RTX 3060 de 6 GB:

```env
MAX_WORKERS=1
OLLAMA_NUM_PARALLEL=1
EMBEDDING_DEVICE=cpu
```

Cada slot paralelo reserva su propio KV cache de `num_ctx` tokens; la VRAM
crece con `OLLAMA_NUM_PARALLEL`. Valores conservadores:

| VRAM | `MAX_WORKERS` |
|---|---|
| 6-8 GB | 1 |
| 12 GB | 2 |
| 16 GB | 3 |
| 24 GB | 4-6 |

`BATCH_SIZE` debe ser mayor o igual que `MAX_WORKERS`. Si Ollama no tiene slots
libres, las peticiones se encolan; no hay aceleración.

Si Ollama corre como servicio systemd, configura `OLLAMA_NUM_PARALLEL` en el
servicio (`systemctl edit ollama`) en lugar de usar `make ollama-start`, y
reinícialo. Si necesitas ahorrar VRAM, puedes probar
`OLLAMA_FLASH_ATTENTION=1` y `OLLAMA_KV_CACHE_TYPE=q8_0` (verifica que tu
versión de Ollama los soporte).

Para ver el comando sin ejecutarlo:

```bash
python scripts/start_ollama.py --num-parallel 3 --dry-run
```

### 4. Configurar embeddings en GPU (opcional)

```env
EMBEDDING_DEVICE=cpu    # cpu | cuda | cuda:N | mps | auto
```

- `cuda` requiere PyTorch con CUDA.
- `auto` elige CUDA si está disponible; si falla por falta de memoria,
  reintenta en CPU.
- En 6 GB conviene `cpu`: Gemma ya ocupa ~3 GB y el embedder compite por VRAM.
- Los embeddings de los 444 códigos ya están precalculados en
  `data/isco08/embeddings/` y los de los títulos se cachean. El cuello de
  botella es la generación de Gemma, no el embedding.

### 5. Verificar la instalación

```bash
make ollama-check
make gpu-check
```

`make gpu-check` muestra la configuración efectiva, la versión de PyTorch,
CUDA, la VRAM libre y los modelos cargados en Ollama.

## Clasificar casos

### Formato de entrada

CSV con dos columnas:

```csv
id,title
1,Recepcionista
2,Jefe de mantenimiento
3,Operario de línea de envasado
```

- `id`: identificador único y no vacío.
- `title`: título no vacío.
- No se usa ninguna otra columna.

Archivo de ejemplo incluido:

```text
data/input/sample10.csv
```

El archivo completo de 1.000 títulos no se incluye en el repositorio. Para
usarlo, pasa tu propio CSV con `INPUT=...`.

### Comandos

```bash
make smoke
make run INPUT=ruta/a/tu_archivo.csv RUN_NAME=mi_corrida
make resume INPUT=ruta/a/tu_archivo.csv RUN_NAME=mi_corrida
```

- `make smoke`: procesa los primeros 10 títulos incluidos.
- `make run`: empieza de cero y borra el checkpoint y las salidas de ese
  `RUN_NAME`.
- `make resume`: continúa desde el checkpoint de ese `RUN_NAME`.

También se pueden ajustar los workers y el tamaño de lote:

```bash
make run INPUT=ruta/a/tu_archivo.csv RUN_NAME=mi_corrida WORKERS=3 BATCH_SIZE=20
```

O directamente:

```bash
python scripts/run.py --input ruta/a/tu_archivo.csv --run-name mi_corrida --workers 3
```

`--workers 0` usa `MAX_WORKERS` del `.env`.

### Salidas

```text
outputs/predictions_<run>.csv
outputs/predictions_<run>.xlsx
outputs/predictions.csv          # copia estable
outputs/predictions.xlsx
outputs/checkpoints/<run>.jsonl
outputs/run_summary_<run>.json
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
- El resumen JSON incluye la configuración efectiva: modelo, workers, batch,
  dispositivo de embeddings, tiempos y uso de tokens.

### Checkpoint

El runner procesa por lotes. Si se interrumpe durante las llamadas de un lote,
ese lote puede repetirse; los lotes ya cerrados no se recalculan. `make resume`
continúa desde el último checkpoint.

## Cómo funciona

### Componentes

- **Embeddings:** `ibm-granite/granite-embedding-311m-multilingual-r2`
  (768 dimensiones; CPU por defecto, configurable a CUDA).
- **Retriever:** distancia coseno contra los 444 códigos CIUO-08.CL.
- **Generador:** `hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S` en Ollama, con
  razonamiento (`<|think|>`).
- **Taxonomía:** 444 grupos primarios, 2.325 ejemplos de cargos.

### Flujo

1. Leer `id,title`.
2. Embeber el título.
3. Calcular la distancia coseno contra los 444 códigos.
4. Ordenar de menor a mayor distancia y recuperar top-20.
5. Enviar los top-10 a Gemma.
6. Parsear el código de 4 dígitos.
7. Validar y reparar por prefijo si es necesario.
8. Guardar checkpoint, CSV, XLSX y resumen.

### Distancia

La distancia usada es:

```text
d = 1 - similitud_coseno
```

Los embeddings están normalizados L2, así que:

```text
d = 1 - (vector_título · vector_código)
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
- En 6 GB conviene `MAX_WORKERS=1` y `EMBEDDING_DEVICE=cpu`: Gemma y el
  embedder compiten por la misma VRAM.

## Licencia

AGPL-3.0. Ver `LICENSE`.
