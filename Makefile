CONFIG       := organism_config.example.yaml
COMPOSE_VLLM := docker compose -f docker-compose.vllm-local.yml
COMPOSE_MON  := docker compose -f monitoring/docker-compose.yml --project-directory monitoring

.PHONY: up down serve proxy bench bench-http logs ps

## Docker: vLLM (8001, 8002) + Prometheus (9090) + Grafana (3000)
up:
	$(COMPOSE_VLLM) up -d
	$(COMPOSE_MON) up -d

down:
	$(COMPOSE_VLLM) down
	$(COMPOSE_MON) down

ps:
	$(COMPOSE_VLLM) ps
	$(COMPOSE_MON) ps

logs:
	$(COMPOSE_VLLM) logs -f

## Organism API server (port 8000)
serve:
	ORGANISM_CONFIG_PATH=$(CONFIG) uvicorn organism.api.server:app --host 0.0.0.0 --port 8000

## Organism Memory Proxy (port 9000) — for Claude Code with full memory (RAG + Tier 2 facts)
proxy:
	ORGANISM_PROXY_CONFIG=organism_proxy.yaml uvicorn organism.proxy.server:app --host 0.0.0.0 --port 9000

## Benchmark — direct mode (loads model in-process, no running server needed)
bench:
	python scripts/bench/longmemeval.py --config $(CONFIG) --workers 1

## Benchmark — HTTP mode (against a running `make serve`)
bench-http:
	python scripts/bench/longmemeval.py --api-url http://localhost:8000

## Tests
test:
	python -m pytest tests/unit/ tests/integration/ -x -q
