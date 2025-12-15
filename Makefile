.PHONY: value-up value-down

# Start all Value services
value-up:
	@if [ ! -f value-engine/.env ] && [ -f value-engine/.env.example ]; then \
		echo "Copying value-engine/.env.example to value-engine/.env"; \
		cp value-engine/.env.example value-engine/.env; \
	fi
	@if [ ! -f value-control-plane-backend/.env ] && [ -f value-control-plane-backend/.env.example ]; then \
		echo "Copying value-control-plane-backend/.env.example to value-control-plane-backend/.env"; \
		cp value-control-plane-backend/.env.example value-control-plane-backend/.env; \
	fi
	docker compose --env-file value-engine/.env \
		-f docker-compose.yml \
		-f value-engine/docker-compose.yml \
		-f value-control-plane-backend/docker-compose.yml \
		-f value-ui/docker-compose.yml up -d --pull always

# Stop all Value services
value-down:
	docker compose --env-file value-engine/.env \
	-f docker-compose.yml \
	-f value-engine/docker-compose.yml \
	-f value-control-plane-backend/docker-compose.yml \
	-f value-ui/docker-compose.yml down

value-clean-volumes: ## Clean Docker volumes
	@read -p "This will stop all Value services and permanently delete Docker volumes (clickhouse, postgres, redis). Are you sure? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(MAKE) value-down; \
		docker volume rm value-private_clickhouse_data || true; \
		docker volume rm value-private_clickhouse_logs || true; \
		docker volume rm value-private_postgres_data || true; \
		docker volume rm value-private_redis_data || true; \
		docker volume rm value-private_value-invoice-data || true; \
	else \
		echo "Aborted volume deletion."; \
	fi