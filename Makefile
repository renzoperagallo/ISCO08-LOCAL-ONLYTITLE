PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
INPUT ?= data/input/empleos_1000_titles.csv
SAMPLE ?= data/input/sample10.csv
BATCH_SIZE ?= 20
RUN_NAME ?= titles
OLLAMA_MODEL ?= hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S

.PHONY: install smoke run resume test ollama-check clean

install:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

smoke:
	$(PY) scripts/run.py --input $(SAMPLE) --batch-size 5 --run-name smoke10 --fresh

run:
	@test -f "$(INPUT)" || (echo "No existe $(INPUT). Pasa INPUT=ruta/a/tu_archivo.csv"; exit 1)
	$(PY) scripts/run.py --input $(INPUT) --batch-size $(BATCH_SIZE) --run-name $(RUN_NAME) --fresh

resume:
	@test -f "$(INPUT)" || (echo "No existe $(INPUT). Pasa INPUT=ruta/a/tu_archivo.csv"; exit 1)
	$(PY) scripts/run.py --input $(INPUT) --batch-size $(BATCH_SIZE) --run-name $(RUN_NAME)

test:
	$(PY) -m pytest -q

ollama-check:
	@curl -s http://127.0.0.1:11434/api/version
	@echo
	@curl -s http://127.0.0.1:11434/api/tags | grep -F "$(OLLAMA_MODEL)" >/dev/null && echo "Modelo disponible: $(OLLAMA_MODEL)" || echo "Modelo no encontrado: $(OLLAMA_MODEL)"

clean:
	rm -rf outputs data/cache .pytest_cache __pycache__ scripts/__pycache__ src/isco08/__pycache__ tests/__pycache__
