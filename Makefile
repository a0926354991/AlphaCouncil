.PHONY: sync lock run run-cli web api-server k8000

sync:
	uv sync

lock:
	uv lock

run:
	uv run adk run alpha_council

run-cli:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	if [ -f alpha_council/.env ]; then set -a; . ./alpha_council/.env; set +a; fi; \
	uv run alpha-council run --ticker 2330 --market tw --masters 1,2,3

web:
	uv run adk web

api-server:
	uv run adk api_server

k8000:
	@pids="$$(lsof -ti :8000)"; \
	if [ -n "$$pids" ]; then \
		kill -9 $$pids; \
		echo "Killed process(es) on :8000 -> $$pids"; \
	else \
		echo "No process is listening on :8000"; \
	fi
