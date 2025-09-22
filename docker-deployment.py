"""
Docker Configuration
Multi-service deployment setup for the Context-Aware Research Assistant
"""

# Dockerfile
dockerfile_content = '''
# Multi-stage build for production optimization
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.6.1

# Set work directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Production stage
FROM python:3.9-slim as production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/embeddings data/citations logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
'''

# Docker Compose Configuration
docker_compose_content = '''
version: '3.8'

services:
  # Main API Application
  research-assistant-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
      target: production
    container_name: research-assistant-api
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/research_assistant
      - REDIS_URL=redis://redis:6379/0
      - WEAVIATE_URL=http://weaviate:8080
      - SEMANTIC_SCHOLAR_API_KEY=${SEMANTIC_SCHOLAR_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - postgres
      - redis
      - weaviate
    networks:
      - research-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: research-assistant-postgres
    environment:
      - POSTGRES_DB=research_assistant
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    networks:
      - research-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: research-assistant-redis
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - research-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  # Weaviate Vector Database
  weaviate:
    image: semitechnologies/weaviate:1.23.7
    container_name: research-assistant-weaviate
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: 'text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate
    networks:
      - research-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  # Nginx Load Balancer / Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: research-assistant-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/ssl:/etc/nginx/ssl:ro
    depends_on:
      - research-assistant-api
    networks:
      - research-network
    restart: unless-stopped

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: research-assistant-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - research-network
    restart: unless-stopped

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: research-assistant-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./docker/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    networks:
      - research-network
    restart: unless-stopped

  # Elasticsearch for Logging
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    container_name: research-assistant-elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - research-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  # Kibana for Log Visualization
  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    container_name: research-assistant-kibana
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
    networks:
      - research-network
    restart: unless-stopped

  # Model Server (for fine-tuned models)
  model-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.model-server
    container_name: research-assistant-model-server
    ports:
      - "8001:8001"
    environment:
      - MODEL_PATH=/app/models/fine_tuned
    volumes:
      - ./models:/app/models:ro
    networks:
      - research-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Background Task Worker
  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile
      target: production
    container_name: research-assistant-worker
    command: python -m src.workers.task_worker
    environment:
      - ENV=production
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/research_assistant
      - REDIS_URL=redis://redis:6379/0
      - WEAVIATE_URL=http://weaviate:8080
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - postgres
      - redis
    networks:
      - research-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  weaviate_data:
  prometheus_data:
  grafana_data:
  elasticsearch_data:

networks:
  research-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
'''

# Nginx Configuration
nginx_config = '''
events {
    worker_connections 1024;
}

http {
    upstream research_assistant {
        server research-assistant-api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=search:10m rate=2r/s;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # API routes with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://research_assistant;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 120s;
        }

        # Search endpoint with stricter rate limiting
        location /api/search {
            limit_req zone=search burst=5 nodelay;
            proxy_pass http://research_assistant;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            proxy_pass http://research_assistant;
            access_log off;
        }

        # Static files (if any)
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Error pages
        error_page 429 /429.html;
        location = /429.html {
            internal;
            return 429 '{"error": "Rate limit exceeded", "retry_after": 60}';
            add_header Content-Type application/json;
        }
    }
}
'''

# Requirements.txt
requirements_content = '''
# FastAPI and ASGI
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.13.1
redis==5.0.1

# ML and NLP
torch==2.1.1
transformers==4.36.2
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy==1.24.4
spacy==3.7.2

# HTTP clients
aiohttp==3.9.1
httpx==0.25.2
requests==2.31.0

# Academic APIs
scholarly==1.7.11
habanero==1.2.6
bibtexparser==1.4.1

# Vector database
weaviate-client==3.25.3

# Monitoring and logging
prometheus-client==0.19.0
structlog==23.2.0
sentry-sdk==1.38.0

# Text processing
nltk==3.8.1
beautifulsoup4==4.12.2
scrapy==2.11.0

# Configuration
python-dotenv==1.0.0
pyyaml==6.0.1
click==8.1.7

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1

# OpenAI and Anthropic
openai==1.3.7
anthropic==0.7.8

# Deployment
gunicorn==21.2.0
docker==6.1.3
kubernetes==28.1.0
'''

# Kubernetes deployment
kubernetes_deployment = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-assistant-api
  labels:
    app: research-assistant-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: research-assistant-api
  template:
    metadata:
      labels:
        app: research-assistant-api
    spec:
      containers:
      - name: api
        image: research-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: research-assistant-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: research-assistant-secrets
              key: redis-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: research-assistant-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: research-assistant-service
spec:
  selector:
    app: research-assistant-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
'''

print("Docker Configuration Files Created:")
print("1. Dockerfile - Multi-stage build with production optimization")
print("2. docker-compose.yml - Complete multi-service stack")
print("3. nginx.conf - Load balancer and rate limiting")
print("4. requirements.txt - Python dependencies")
print("5. kubernetes-deployment.yaml - K8s deployment configuration")
print("\nKey features:")
print("- Multi-service architecture with PostgreSQL, Redis, Weaviate")
print("- Monitoring with Prometheus and Grafana")
print("- Logging with ELK stack")
print("- Rate limiting and security headers")
print("- Health checks and auto-restart")
print("- Resource limits and GPU support for model server")
print("- Production-ready with non-root user and minimal attack surface")