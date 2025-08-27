# BEM Production Deployment Guide

This guide provides comprehensive instructions for deploying BEM (Basis Extension Modules) in production environments, from small-scale applications to enterprise-grade systems.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Environment Preparation](#environment-preparation)
- [Container Deployment](#container-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Scaling and Load Balancing](#scaling-and-load-balancing)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Configuration](#security-configuration)
- [Performance Optimization](#performance-optimization)
- [Maintenance and Updates](#maintenance-and-updates)
- [Troubleshooting](#troubleshooting)

## Deployment Overview

BEM supports multiple deployment patterns optimized for different use cases and scales.

### Deployment Options

| Pattern | Use Case | Scale | Complexity |
|---------|----------|-------|------------|
| **Single Instance** | Development, small applications | 1-100 requests/min | Low |
| **Container Deployment** | Production services, microservices | 100-1000 requests/min | Medium |
| **Kubernetes** | High availability, auto-scaling | 1000+ requests/min | High |
| **Cloud Managed** | Serverless, managed infrastructure | Variable | Medium |
| **Edge Deployment** | Low latency, offline capability | 10-100 requests/min | Medium |

### Architecture Components

```mermaid
graph TD
    A[Load Balancer] --> B[API Gateway]
    B --> C[BEM Service Instances]
    C --> D[Model Storage]
    C --> E[Configuration Service]
    C --> F[Monitoring System]
    C --> G[Logging Aggregator]
    
    H[Redis Cache] --> C
    I[Database] --> C
    J[GPU Pool] --> C
```

## Environment Preparation

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores, 2.0+ GHz
- **Memory**: 8 GB RAM
- **Storage**: 20 GB SSD
- **GPU**: Optional (CUDA-compatible for acceleration)
- **Network**: 100 Mbps
- **OS**: Ubuntu 20.04+, CentOS 8+, or compatible Linux

#### Production Requirements (Single Instance)
- **CPU**: 8 cores, 3.0+ GHz
- **Memory**: 32 GB RAM  
- **Storage**: 100 GB SSD
- **GPU**: NVIDIA RTX 3080+ or T4+ for GPU acceleration
- **Network**: 1 Gbps
- **OS**: Ubuntu 22.04 LTS (recommended)

#### Enterprise Requirements (Multi-Instance)
- **CPU**: 16+ cores per instance, 3.0+ GHz
- **Memory**: 64+ GB RAM per instance
- **Storage**: 500+ GB NVMe SSD
- **GPU**: NVIDIA A100, V100, or H100 for high-performance workloads
- **Network**: 10+ Gbps with load balancing
- **Container Orchestration**: Kubernetes cluster

### Software Prerequisites

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+
sudo apt install python3.9 python3.9-pip python3.9-venv

# Install system dependencies
sudo apt install build-essential git curl wget \
                 nginx redis-server postgresql-14 \
                 docker.io docker-compose

# Install NVIDIA drivers and CUDA (for GPU support)
sudo apt install nvidia-driver-530 nvidia-cuda-toolkit

# Verify installations
python3.9 --version
docker --version
nvidia-smi  # Verify GPU setup
```

### Environment Setup

```bash
# Create deployment user
sudo useradd -m -s /bin/bash bemuser
sudo usermod -aG docker bemuser

# Create deployment directories
sudo mkdir -p /opt/bem/{config,logs,models,data}
sudo chown -R bemuser:bemuser /opt/bem

# Switch to deployment user
sudo su - bemuser

# Clone BEM repository
cd /opt/bem
git clone https://github.com/nathanrice/BEM.git
cd BEM

# Create production virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn[gevent] uvicorn[standard]
```

## Container Deployment

### Docker Configuration

#### Production Dockerfile

```dockerfile
# production.Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV BEM_ENV=production

# Create non-root user
RUN groupadd -r bem && useradd -r -g bem bem

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p logs data checkpoints

# Change ownership to bem user
RUN chown -R bem:bem /app

# Switch to non-root user
USER bem

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python scripts/health_check.py

# Expose port
EXPOSE 8000

# Default command
CMD ["gunicorn", "--config", "config/gunicorn.conf.py", "src.bem_api:app"]
```

#### Docker Compose Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  bem-api:
    build:
      context: .
      dockerfile: production.Dockerfile
    ports:
      - "8000:8000"
    environment:
      - BEM_ENV=production
      - DATABASE_URL=postgresql://bem:${DB_PASSWORD}@postgres:5432/bem_prod
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - model-cache:/app/models/cache
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "scripts/health_check.py"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - bem-api
    restart: unless-stopped

  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=bem_prod
      - POSTGRES_USER=bem
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    restart: unless-stopped

  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  grafana-data:
  model-cache:
```

#### Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream bem_api {
        server bem-api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://bem_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 300s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 128k;
            proxy_buffers 4 256k;
            proxy_busy_buffers_size 256k;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://bem_api/health;
            access_log off;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### Deployment Commands

```bash
# Create environment file
cat > .env << EOF
DB_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
BEM_SECRET_KEY=$(openssl rand -base64 64)
EOF

# Build and deploy
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs bem-api

# Test API endpoint
curl -X GET http://localhost/health
```

## Cloud Deployment

### AWS Deployment

#### ECS Service Configuration

```json
{
  "family": "bem-api-task",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "bem-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/bem-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "BEM_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:bem/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bem-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python scripts/health_check.py || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### CloudFormation Template (excerpt)

```yaml
# cloudformation/bem-infrastructure.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'BEM API Infrastructure'

Resources:
  BEMCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: bem-cluster
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT

  BEMService:
    Type: AWS::ECS::Service
    Properties:
      Cluster: !Ref BEMCluster
      TaskDefinition: !Ref BEMTaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          Subnets:
            - !Ref PrivateSubnet1
            - !Ref PrivateSubnet2
          SecurityGroups:
            - !Ref BEMSecurityGroup
      LoadBalancers:
        - ContainerName: bem-api
          ContainerPort: 8000
          TargetGroupArn: !Ref BEMTargetGroup

  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: bem-alb
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref ALBSecurityGroup

  BEMDatabase:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: bem-postgres
      DBInstanceClass: db.t3.medium
      Engine: postgres
      EngineVersion: '14.6'
      AllocatedStorage: 100
      DBName: bem_prod
      MasterUsername: bem_admin
      MasterUserPassword: !Ref DatabasePassword
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      DBSubnetGroupName: !Ref DatabaseSubnetGroup
```

### Google Cloud Platform

```yaml
# gcp/cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/bem-api:$BUILD_ID', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/bem-api:$BUILD_ID']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'bem-api'
      - '--image=gcr.io/$PROJECT_ID/bem-api:$BUILD_ID'
      - '--region=us-central1'
      - '--platform=managed'
      - '--memory=4Gi'
      - '--cpu=2'
      - '--max-instances=10'
      - '--port=8000'
      - '--set-env-vars=BEM_ENV=production'
```

### Azure Container Instances

```yaml
# azure/container-group.yml
apiVersion: 2019-12-01
location: eastus
name: bem-api-group
properties:
  containers:
  - name: bem-api
    properties:
      image: your-registry.azurecr.io/bem-api:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: BEM_ENV
        value: production
      - name: DATABASE_URL
        secureValue: postgresql://user:pass@server:5432/db
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
```

## Scaling and Load Balancing

### Horizontal Pod Autoscaler (Kubernetes)

```yaml
# k8s/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bem-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bem-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

### Load Balancing Configuration

```python
# config/load_balancer.py
LOAD_BALANCER_CONFIG = {
    "algorithm": "least_connections",  # round_robin, ip_hash, least_connections
    "health_check": {
        "enabled": True,
        "interval": 30,  # seconds
        "timeout": 5,    # seconds
        "path": "/health",
        "expected_status": 200
    },
    "sticky_sessions": False,
    "connection_pooling": {
        "enabled": True,
        "pool_size": 20,
        "max_connections": 1000,
        "keepalive_timeout": 60
    },
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 1000,
        "burst_size": 50
    }
}
```

## Monitoring and Observability

### Prometheus Metrics

```python
# src/bem_api/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
request_count = Counter('bem_requests_total', 'Total API requests', ['method', 'endpoint'])
request_duration = Histogram('bem_request_duration_seconds', 'Request duration')
model_inference_time = Histogram('bem_inference_duration_seconds', 'Model inference time')
active_connections = Gauge('bem_active_connections', 'Active connections')
memory_usage = Gauge('bem_memory_usage_bytes', 'Memory usage in bytes')
gpu_utilization = Gauge('bem_gpu_utilization_percent', 'GPU utilization percentage')

# Safety metrics
safety_violations = Counter('bem_safety_violations_total', 'Safety violations', ['type'])
routing_decisions = Counter('bem_routing_decisions_total', 'Routing decisions', ['expert'])
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "BEM API Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(bem_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(bem_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(bem_requests_total{status=~\"4..|5..\"}[5m]) / rate(bem_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```python
# config/logging.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
            
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': JSONFormatter,
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/bem-api.log',
            'formatter': 'json',
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 5,
            'level': 'DEBUG',
        },
    },
    'loggers': {
        'bem': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'uvicorn': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
```

## Security Configuration

### Environment Variables

```bash
# .env.production
# Application settings
BEM_ENV=production
BEM_DEBUG=false
BEM_LOG_LEVEL=INFO

# Security settings
BEM_SECRET_KEY=your-secret-key-here
BEM_API_KEY_REQUIRED=true
BEM_RATE_LIMIT_ENABLED=true
BEM_CORS_ENABLED=false

# Database configuration
DATABASE_URL=postgresql://user:pass@host:port/database
DATABASE_POOL_SIZE=20
DATABASE_MAX_CONNECTIONS=100

# Cache configuration
REDIS_URL=redis://host:port/0
REDIS_PASSWORD=your-redis-password

# Model configuration
MODEL_CACHE_DIR=/app/models/cache
MODEL_DOWNLOAD_TIMEOUT=300

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
HEALTH_CHECK_ENABLED=true

# Resource limits
MAX_REQUEST_SIZE=10MB
REQUEST_TIMEOUT=300
WORKER_TIMEOUT=600
```

### Security Headers

```python
# src/bem_api/security.py
from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

def configure_security(app: FastAPI):
    # Trusted hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )
    
    # CORS (if needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key="your-session-secret-key"
    )
    
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
```

## Performance Optimization

### Gunicorn Configuration

```python
# config/gunicorn.conf.py
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeouts
timeout = 120
keepalive = 2
graceful_timeout = 30

# Process naming
proc_name = 'bem-api'

# Logging
loglevel = 'info'
accesslog = '/app/logs/access.log'
errorlog = '/app/logs/error.log'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process management
preload_app = True
daemon = False
user = 1000
group = 1000

# Performance tuning
worker_tmp_dir = '/dev/shm'  # Use RAM for temporary files
```

### Redis Caching Strategy

```python
# src/bem_api/cache.py
import redis
import json
import hashlib
from typing import Optional, Any

class BEMCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        
    def _generate_key(self, prefix: str, data: dict) -> str:
        """Generate cache key from input data."""
        key_data = json.dumps(data, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"bem:{prefix}:{key_hash}"
    
    def cache_model_output(self, inputs: dict, output: dict, ttl: int = 3600):
        """Cache model inference output."""
        key = self._generate_key("inference", inputs)
        self.redis.setex(key, ttl, json.dumps(output))
    
    def get_cached_output(self, inputs: dict) -> Optional[dict]:
        """Retrieve cached model output."""
        key = self._generate_key("inference", inputs)
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None
    
    def cache_routing_decision(self, context: dict, routing: dict, ttl: int = 1800):
        """Cache routing decisions for similar contexts."""
        key = self._generate_key("routing", context)
        self.redis.setex(key, ttl, json.dumps(routing))
```

## Maintenance and Updates

### Update Strategy

```bash
#!/bin/bash
# scripts/update_deployment.sh

set -e

# Configuration
ENVIRONMENT=${1:-production}
NEW_VERSION=${2:-latest}
HEALTH_CHECK_URL="http://localhost/health"

echo "Starting BEM API update to version $NEW_VERSION"

# Pull new image
docker pull your-registry/bem-api:$NEW_VERSION

# Create backup of current state
docker-compose -f docker-compose.${ENVIRONMENT}.yml exec postgres \
    pg_dump -U bem bem_prod > backups/backup_$(date +%Y%m%d_%H%M%S).sql

# Update with rolling deployment
docker-compose -f docker-compose.${ENVIRONMENT}.yml up -d --scale bem-api=4
sleep 30

# Health check
for i in {1..10}; do
    if curl -f $HEALTH_CHECK_URL; then
        echo "Health check passed"
        break
    fi
    echo "Health check failed, attempt $i/10"
    sleep 10
done

# Scale down old instances
docker-compose -f docker-compose.${ENVIRONMENT}.yml up -d --scale bem-api=2

echo "Update completed successfully"
```

### Backup and Recovery

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/opt/bem/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -h postgres -U bem bem_prod | gzip > $BACKUP_DIR/database.sql.gz

# Model checkpoints backup
tar -czf $BACKUP_DIR/models.tar.gz /app/models/

# Configuration backup
cp -r /app/config/ $BACKUP_DIR/config/

# Upload to cloud storage (example with AWS S3)
aws s3 cp $BACKUP_DIR s3://your-backup-bucket/bem-backups/$(date +%Y%m%d)/ --recursive

# Clean up old local backups (keep 7 days)
find /opt/bem/backups/ -type d -mtime +7 -exec rm -rf {} +
```

## Troubleshooting

### Common Issues and Solutions

#### High Memory Usage

```bash
# Monitor memory usage
docker stats bem-api

# Check for memory leaks
docker exec bem-api python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Restart if memory usage is too high
if [ $(docker stats --no-stream --format "{{.MemUsage}}" bem-api | cut -d'/' -f1 | cut -d'G' -f1) -gt 3 ]; then
    docker-compose restart bem-api
fi
```

#### Database Connection Issues

```sql
-- Check database connections
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';

-- Kill long-running queries
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'active' 
AND query_start < NOW() - INTERVAL '5 minutes';
```

#### Model Loading Failures

```python
# Model loading diagnostics
import torch
import psutil

def diagnose_model_loading():
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    print(f"CPU Memory: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"CPU Memory Used: {psutil.virtual_memory().used / 1e9:.1f} GB")
```

### Performance Debugging

```bash
# Profile API performance
curl -X POST http://localhost/profile \
  -H "Content-Type: application/json" \
  -d '{"text": "test input", "profile": true}'

# Monitor GPU utilization
nvidia-smi --loop=1

# Check API response times
siege -c 10 -t 1M http://localhost/api/generate

# Database performance monitoring
docker exec postgres pg_stat_statements_reset();
# ... run load test ...
docker exec postgres psql -U bem -d bem_prod -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
"
```

### Log Analysis

```bash
# Analyze error patterns
grep "ERROR" /app/logs/bem-api.log | jq -r '.message' | sort | uniq -c | sort -nr

# Monitor request rates
grep "INFO" /app/logs/bem-api.log | jq -r '.timestamp' | cut -c1-16 | sort | uniq -c

# Safety violation analysis
grep "safety_violation" /app/logs/bem-api.log | jq -r '.violation_type' | sort | uniq -c
```

---

**Deployment Guide Version**: 1.0  
**Last Updated**: December 2024  
**Compatible with BEM**: 2.0.0+

For deployment support and enterprise consulting, please contact the BEM team through GitHub Issues or email.