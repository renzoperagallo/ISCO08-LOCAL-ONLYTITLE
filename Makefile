PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
INPUT ?= data/input/empleos_1000_titles.csv
SAMPLE ?= data/input/sample10.csv
BATCH_SIZE ?= 20
WORKERS ?= 0
NUM_PARALLEL ?= 0
RUN_NAME ?= titles
OLLAMA_MODEL ?= hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S

.PHONY: install smoke run resume test ollama-check ollama-start ollama-start-bg gpu-check clean

install:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

smoke:
	$(PY) scripts/run.py --input $(SAMPLE) --batch-size 5 --workers $(WORKERS) --run-name smoke10 --fresh

run:
	@test -f "$(INPUT)" || (echo "No existe $(INPUT). Pasa INPUT=ruta/a/tu_archivo.csv"; exit 1)
	$(PY) scripts/run.py --input $(INPUT) --batch-size $(BATCH_SIZE) --workers $(WORKERS) --run-name $(RUN_NAME) --fresh

resume:
	@test -f "$(INPUT)" || (echo "No existe $(INPUT). Pasa INPUT=ruta/a/tu_archivo.csv"; exit 1)
	$(PY) scripts/run.py --input $(INPUT) --batch-size $(BATCH_SIZE) --workers $(WORKERS) --run-name $(RUN_NAME)

test:
	$(PY) -m pytest -q

ollama-check:
	@curl -s http://127.0.0.1:11434/api/version
	@echo
	@curl -s http://127.0.0.1:11434/api/tags | grep -F "$(OLLAMA_MODEL)" >/dev/null && echo "Modelo disponible: $(OLLAMA_MODEL)" || echo "Modelo no encontrado: $(OLLAMA_MODEL)"

ollama-start:
	$(PY) scripts/start_ollama.py --num-parallel $(NUM_PARALLEL)

ollama-start-bg:
	@mkdir -p logs
	@rm -f logs/ollama.pid
	@nohup $(PY) scripts/start_ollama.py --num-parallel $(NUM_PARALLEL) > logs/ollama.log 2>&1 & echo $$! > logs/ollama.pid
	@sleep 1; \
	if kill -0 $$(cat logs/ollama.pid) 2>/dev/null; then \
		echo "Ollama en background. PID: $$(cat logs/ollama.pid) | log: logs/ollama.log"; \
	else \
		echo "Ollama no quedo corriendo. Ultimas lineas del log:"; \
		tail -20 logs/ollama.log; \
		exit 1; \
	fi

gpu-check:
	$(PY) scripts/check_gpu.py

clean:
	rm -rf outputs data/cache logs .pytest_cache __pycache__ scripts/__pycache__ src/isco08/__pycache__ tests/__pycache__
