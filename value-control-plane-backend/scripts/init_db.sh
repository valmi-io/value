#!/bin/bash
# Validate required environment variables
if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_DB" ]; then
    echo "Error: Required environment variables not set!"
    echo "Please ensure POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB are defined in .env"
    exit 1
fi

# Determine the superuser
SUPERUSER=${POSTGRES_USER:-postgres}
# Set the host to connect to the other container
POSTGRES_HOST=${POSTGRES_HOST:-value-db}

echo "Using superuser: $SUPERUSER"
echo "Connecting to host: $POSTGRES_HOST"
echo "PostgreSQL is ready!"

# Check if database exists
echo "Checking if database '$POSTGRES_DB' exists..."
DB_EXISTS=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB';" 2>/dev/null || echo "")

if [ -z "$DB_EXISTS" ]; then
    echo "Creating database '$POSTGRES_DB'..."
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d postgres -c "CREATE DATABASE \"$POSTGRES_DB\";"
    
    echo "Granting all privileges on database '$POSTGRES_DB' to user '$POSTGRES_USER'..."
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE \"$POSTGRES_DB\" TO \"$POSTGRES_USER\";"
    
    # Connect to the new database and grant schema privileges
    echo "Granting schema privileges..."
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d "$POSTGRES_DB" -c "GRANT ALL ON SCHEMA public TO \"$POSTGRES_USER\";"
    
    # Grant default privileges for future tables
    echo "Setting default privileges for future objects..."
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d "$POSTGRES_DB" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO \"$POSTGRES_USER\";"
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d "$POSTGRES_DB" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO \"$POSTGRES_USER\";"
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d "$POSTGRES_DB" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO \"$POSTGRES_USER\";"
    
    echo "All privileges granted successfully!"
else
    echo "Database '$POSTGRES_DB' already exists."
    echo "Ensuring user '$POSTGRES_USER' has all privileges..."
    
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE \"$POSTGRES_DB\" TO \"$POSTGRES_USER\";"
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d "$POSTGRES_DB" -c "GRANT ALL ON SCHEMA public TO \"$POSTGRES_USER\";"
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d "$POSTGRES_DB" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO \"$POSTGRES_USER\";"
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d "$POSTGRES_DB" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO \"$POSTGRES_USER\";"
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $SUPERUSER -d "$POSTGRES_DB" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO \"$POSTGRES_USER\";"
    
    echo "Privileges updated successfully!"
fi

echo "Database initialization complete!"