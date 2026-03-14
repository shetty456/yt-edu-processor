.PHONY: install dev run docker-build docker-run docker-stop health

PORT ?= 8000

## Install dependencies into the active virtualenv / system Python
install:
	pip install -r requirements.txt

## Run with hot-reload (development)
dev:
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

## Run without hot-reload (production-like, mirrors Dockerfile CMD)
run:
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

## Copy .env.example → .env if .env doesn't exist yet
.env:
	cp .env.example .env
	@echo ".env created — fill in your API keys before running."

## Build the Docker image
docker-build:
	docker build -t yt-edu-processor .

## Run the Docker container (loads .env automatically)
docker-run: docker-build
	docker run --rm -p $(PORT):$(PORT) --env-file .env -e PORT=$(PORT) yt-edu-processor

## Stop any running container for this image
docker-stop:
	docker ps -q --filter ancestor=yt-edu-processor | xargs -r docker stop

## Hit the health endpoint to confirm the server is up
health:
	curl -sf http://localhost:$(PORT)/health && echo " ✓ healthy" || echo " ✗ not reachable"
