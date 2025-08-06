# -------- Start production stack -------- 
# make prod
# -------- View logs -------- 
# make prod-logs
# -------- Health check -------- 
# make health
# -------- Stop services -------- 
# make stop

.PHONY: prod dev stop clean logs health

# Production commands
prod:
	docker-compose -f docker-compose.prod.yml --env-file .env.prod up

prod-build:
	docker-compose -f docker-compose.prod.yml --env-file .env.prod up --build --scale worker=2

prod-logs:
	docker-compose -f docker-compose.prod.yml logs -f

# Development commands
dev:
	docker-compose -f docker-compose.dev.yml up -d

dev-build:
	docker-compose -f docker-compose.dev.yml up -d --build

dev-logs:
	docker-compose -f docker-compose.dev.yml logs -f

# Common commands
stop:
	docker-compose -f docker-compose.prod.yml down
	docker-compose -f docker-compose.dev.yml down

clean:
	docker-compose -f docker-compose.prod.yml down -v
	docker-compose -f docker-compose.dev.yml down -v
	docker system prune -f

health:
	@echo "=== Service Health Check ==="
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo "\n=== Redis Health ==="
	@docker exec shadow-trainer-redis redis-cli ping || echo "Redis not responding"
	@echo "\n=== API Health ==="
	@curl -s http://localhost:8000/health || echo "API not responding"

logs:
	docker-compose -f docker-compose.prod.yml logs --tail=100

restart:
	make stop
	make prod