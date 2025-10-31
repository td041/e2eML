build: 
	docker compose up -d --build
run:
	docker compose up -d
down:
	docker compose down -v

.PHONY: build down run