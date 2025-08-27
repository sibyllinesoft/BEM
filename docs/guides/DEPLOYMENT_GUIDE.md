# BEM v1.3 Production Deployment Guide

## 🚀 Overview

This deployment guide provides comprehensive instructions for deploying BEM v1.3 Performance+Agentic Sprint system in production environments. The system includes advanced safety monitoring, performance optimization, and operational resilience features for enterprise deployment.

## 📋 Table of Contents

1. [Production Architecture](#production-architecture)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring and Observability](#monitoring-and-observability)
5. [Safety and Rollback Systems](#safety-and-rollback-systems)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Disaster Recovery](#disaster-recovery)

## 🏗️ Production Architecture

### System Components Overview

```
Production BEM v1.3 Deployment Architecture
├── Load Balancer (nginx/HAProxy)
│   ├── SSL Termination
│   ├── Request routing
│   └── Health checks
├── BEM Application Cluster
│   ├── Primary Instance (Active experiments)
│   ├── Shadow Instance (A/B testing)  
│   └── Fallback Instance (Safety rollback)
├── Monitoring Stack
│   ├── Metrics Collection (Prometheus)
│   ├── Logging Aggregation (ELK Stack)
│   ├── Alerting (AlertManager)
│   └── Dashboards (Grafana)
├── Data Storage
│   ├── Experiment Results (PostgreSQL)
│   ├── Model Artifacts (S3/MinIO)
│   ├── Logs (Elasticsearch)
│   └── Metrics (InfluxDB)
└── Safety Systems
    ├── Drift Detection
    ├── Performance Gates
    ├── Circuit Breakers
    └── Auto-rollback
```

### Deployment Patterns

**1. Blue-Green Deployment**:
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  bem-blue:
    image: bem-v13:latest
    environment:
      - DEPLOYMENT_SLOT=blue
      - BEM_CONFIG_PATH=/config/production.yml
    volumes:
      - ./config:/config:ro
      - ./models:/models:ro
    ports:
      - "8000:8000"
      
  bem-green:
    image: bem-v13:latest  
    environment:
      - DEPLOYMENT_SLOT=green
      - BEM_CONFIG_PATH=/config/production.yml
    volumes:
      - ./config:/config:ro
      - ./models:/models:ro
    ports:
      - "8001:8000"
      
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - bem-blue
      - bem-green
```

**2. Canary Deployment**:
```python
# bem2/deployment/canary_controller.py
class CanaryController:
    """
    Canary deployment controller for gradual rollout.
    Implements progressive traffic shifting with safety gates.
    """
    
    def __init__(self, config: CanaryConfig):
        self.config = config
        self.traffic_splitter = TrafficSplitter()
        self.safety_monitor = SafetyMonitor()
        self.rollback_manager = RollbackManager()
        
    def deploy_canary(self, new_version: str):
        """Deploy new version with progressive rollout"""
        # Phase 1: 5% traffic to canary
        self.traffic_splitter.set_weights(canary=0.05, stable=0.95)
        self._monitor_phase(duration=300, phase="initial")
        
        # Phase 2: 25% traffic if metrics healthy
        if self.safety_monitor.is_healthy():
            self.traffic_splitter.set_weights(canary=0.25, stable=0.75)
            self._monitor_phase(duration=600, phase="ramp")
        else:
            self.rollback_manager.rollback_canary()
            return DeploymentResult.FAILED
            
        # Phase 3: 100% traffic if still healthy
        if self.safety_monitor.is_healthy():
            self.traffic_splitter.set_weights(canary=1.0, stable=0.0)
            self._monitor_phase(duration=900, phase="full")
        else:
            self.rollback_manager.rollback_canary()
            return DeploymentResult.FAILED
            
        return DeploymentResult.SUCCESS
        
    def _monitor_phase(self, duration: int, phase: str):
        """Monitor deployment phase with safety checks"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metrics = self.safety_monitor.get_current_metrics()
            
            # Check safety thresholds
            if not self.safety_monitor.check_thresholds(metrics):
                logger.error(f"Safety threshold violation in {phase} phase")
                self.rollback_manager.rollback_canary()
                raise DeploymentSafetyError(f"Safety violation in {phase}")
                
            time.sleep(30)  # Check every 30 seconds
```

## 🔧 Infrastructure Requirements

### Hardware Specifications

**Minimum Production Requirements**:
```yaml
compute_nodes:
  cpu: "16 cores (Intel Xeon or AMD EPYC)"
  memory: "64GB RAM"
  gpu: "NVIDIA RTX 4090 or A100 (24GB VRAM)"
  storage: "1TB NVMe SSD"
  network: "10Gbps ethernet"
  
load_balancer:
  cpu: "8 cores"
  memory: "16GB RAM"
  storage: "100GB SSD"
  network: "10Gbps ethernet"
  
monitoring_stack:
  cpu: "8 cores"
  memory: "32GB RAM"
  storage: "2TB SSD for logs/metrics"
  network: "1Gbps ethernet"
```

**Recommended Production Configuration**:
```yaml
compute_cluster:
  primary_nodes: 3
  shadow_nodes: 2
  fallback_nodes: 1
  
scaling_parameters:
  cpu_threshold: "70%"
  memory_threshold: "80%"  
  gpu_utilization_threshold: "85%"
  response_time_threshold: "200ms"
  
redundancy:
  multi_az: true
  backup_frequency: "hourly"
  disaster_recovery_rto: "5 minutes"
  disaster_recovery_rpo: "1 hour"
```

### Container Configuration

**Production Dockerfile**:
```dockerfile
# Multi-stage production build
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip3 install --no-cache-dir -r requirements-prod.txt

# Build custom CUDA kernels
COPY bem2/kernels/ ./kernels/
RUN cd kernels && python3 build_kernels.py --optimize

# Production runtime stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /kernels/build/*.so /app/kernels/
COPY bem2/ /app/bem2/
COPY config/ /app/config/

# Set up production user (non-root)
RUN useradd -m -u 1000 bemuser && chown -R bemuser:bemuser /app
USER bemuser

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import bem2; bem2.health_check()" || exit 1

# Production entrypoint
EXPOSE 8000
CMD ["python3", "-m", "bem2.server", "--config", "/app/config/production.yml"]
```

### Kubernetes Deployment

**Production Kubernetes Manifests**:
```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bem-v13-deployment
  labels:
    app: bem-v13
    version: v1.3.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bem-v13
  template:
    metadata:
      labels:
        app: bem-v13
        version: v1.3.0
    spec:
      containers:
      - name: bem-v13
        image: bem-v13:v1.3.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "8" 
            memory: "32Gi"
            nvidia.com/gpu: "1"
        env:
        - name: BEM_CONFIG_PATH
          value: "/config/production.yml"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "max_split_size_mb:128"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        - name: models
          mountPath: /models
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: bem-config
      - name: models
        persistentVolumeClaim:
          claimName: bem-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: bem-v13-service
spec:
  selector:
    app: bem-v13
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🔧 Deployment Procedures

### Automated Deployment Pipeline

**CI/CD Pipeline** (GitHub Actions):
```yaml
# .github/workflows/deploy.yml
name: BEM v1.3 Production Deployment

on:
  push:
    tags:
      - 'v*'

jobs:
  test:
    runs-on: ubuntu-latest-gpu
    steps:
    - uses: actions/checkout@v3
    
    - name: Run comprehensive tests
      run: |
        python -m pytest tests/ --cov=bem2 --cov-fail-under=85
        python validate_structure.py --comprehensive
        
    - name: Performance benchmarking
      run: |
        python bem_profile.py --all-variants --production-check
        
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t bem-v13:${{ github.ref_name }} .
        docker tag bem-v13:${{ github.ref_name }} bem-v13:latest
        
    - name: Push to registry
      run: |
        docker push bem-v13:${{ github.ref_name }}
        docker push bem-v13:latest
        
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - name: Deploy to staging
      run: |
        kubectl set image deployment/bem-v13-staging bem-v13=bem-v13:${{ github.ref_name }}
        kubectl rollout status deployment/bem-v13-staging
        
    - name: Run staging validation
      run: |
        python tests/test_staging_deployment.py --endpoint $STAGING_URL
        
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
    - name: Deploy with canary
      run: |
        python deployment/canary_deployment.py \
          --new-version ${{ github.ref_name }} \
          --traffic-increment 0.05,0.25,1.0 \
          --safety-thresholds config/safety_thresholds.yml
```

### Manual Deployment Steps

**Step-by-Step Production Deployment**:
```bash
# 1. Pre-deployment validation
python validate_structure.py --production-ready
python test_bem_v13_comprehensive.py --production-mode

# 2. Build and tag container
docker build -t bem-v13:v1.3.0 .
docker tag bem-v13:v1.3.0 registry.company.com/bem-v13:v1.3.0

# 3. Push to registry  
docker push registry.company.com/bem-v13:v1.3.0

# 4. Update configuration
kubectl create configmap bem-config --from-file=config/production.yml

# 5. Deploy with rolling update
kubectl set image deployment/bem-v13-deployment \
  bem-v13=registry.company.com/bem-v13:v1.3.0

# 6. Monitor rollout
kubectl rollout status deployment/bem-v13-deployment --timeout=600s

# 7. Validate deployment
python deployment/validate_production.py --endpoint https://bem.company.com

# 8. Enable traffic gradually (canary)
python deployment/canary_controller.py --enable --version v1.3.0
```

## 📊 Monitoring and Observability

### Metrics Collection

**Prometheus Configuration**:
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "bem_alerts.yml"

scrape_configs:
  - job_name: 'bem-v13'
    static_configs:
      - targets: ['bem-v13-service:8000']
    scrape_interval: 10s
    metrics_path: /metrics
    
  - job_name: 'bem-gpu'
    static_configs:
      - targets: ['bem-v13-service:8001']  
    scrape_interval: 30s
    metrics_path: /gpu-metrics

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

**Key Metrics to Monitor**:
```python
# bem2/monitoring/metrics.py
class ProductionMetrics:
    """Production metrics collection for BEM v1.3"""
    
    def __init__(self):
        # Request metrics
        self.request_duration = Histogram(
            'bem_request_duration_seconds',
            'Time spent processing requests',
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.request_count = Counter(
            'bem_requests_total', 
            'Total requests processed',
            ['method', 'endpoint', 'status']
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'bem_model_accuracy',
            'Current model accuracy',
            ['variant', 'metric_type']
        )
        
        self.inference_latency = Histogram(
            'bem_inference_latency_seconds',
            'Model inference latency',
            ['variant']
        )
        
        # Resource utilization
        self.gpu_utilization = Gauge(
            'bem_gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
        self.memory_usage = Gauge(
            'bem_memory_usage_bytes',
            'Memory usage in bytes',
            ['type']  # 'cpu', 'gpu'
        )
        
        # Safety metrics
        self.safety_violations = Counter(
            'bem_safety_violations_total',
            'Safety violations detected',
            ['violation_type']
        )
        
        self.drift_alerts = Counter(
            'bem_drift_alerts_total', 
            'Model drift alerts',
            ['component']
        )
        
    def record_request(self, duration, method, endpoint, status):
        """Record request metrics"""
        self.request_duration.observe(duration)
        self.request_count.labels(
            method=method, 
            endpoint=endpoint, 
            status=status
        ).inc()
        
    def update_model_metrics(self, variant, accuracy, latency):
        """Update model performance metrics"""
        self.model_accuracy.labels(
            variant=variant, 
            metric_type='accuracy'
        ).set(accuracy)
        
        self.inference_latency.labels(variant=variant).observe(latency)
```

### Alerting Rules

**Alert Configuration**:
```yaml
# bem_alerts.yml
groups:
- name: bem_alerts
  rules:
  # High error rate
  - alert: BEMHighErrorRate
    expr: |
      (
        sum(rate(bem_requests_total{status=~"5.."}[5m])) /
        sum(rate(bem_requests_total[5m]))
      ) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "BEM error rate is above 5%"
      description: "Error rate is {{ $value | humanizePercentage }}"
      
  # High response time
  - alert: BEMHighLatency
    expr: |
      histogram_quantile(0.95, 
        sum(rate(bem_request_duration_seconds_bucket[5m])) by (le)
      ) > 2.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "BEM 95th percentile latency is high"
      description: "95th percentile latency is {{ $value }}s"
      
  # GPU utilization
  - alert: BEMHighGPUUtilization
    expr: bem_gpu_utilization_percent > 90
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "BEM GPU utilization is high"
      description: "GPU utilization is {{ $value }}%"
      
  # Safety violations
  - alert: BEMSafetyViolation
    expr: increase(bem_safety_violations_total[5m]) > 0
    for: 0s
    labels:
      severity: critical
    annotations:
      summary: "BEM safety violation detected"
      description: "Safety violation of type {{ $labels.violation_type }}"
      
  # Model drift
  - alert: BEMModelDrift
    expr: increase(bem_drift_alerts_total[15m]) > 2
    for: 0s
    labels:
      severity: warning  
    annotations:
      summary: "BEM model drift detected"
      description: "Drift detected in {{ $labels.component }}"
```

### Dashboard Configuration

**Grafana Dashboard** (JSON extract):
```json
{
  "dashboard": {
    "title": "BEM v1.3 Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(bem_requests_total[5m]))",
            "legendFormat": "Total RPS"
          }
        ]
      },
      {
        "title": "Response Time Percentiles",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(bem_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(bem_request_duration_seconds_bucket[5m])) by (le))", 
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(bem_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "bem_model_accuracy",
            "legendFormat": "Accuracy - {{ variant }}"
          }
        ]
      }
    ]
  }
}
```

## 🛡️ Safety and Rollback Systems

### Circuit Breaker Implementation

**Production Circuit Breaker**:
```python
# bem2/safety/circuit_breaker.py
class ProductionCircuitBreaker:
    """
    Circuit breaker for production safety.
    Prevents cascading failures and enables graceful degradation.
    """
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
            
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker reset to CLOSED")
            
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
    def _should_attempt_reset(self):
        """Check if we should attempt to reset the circuit breaker"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
```

### Automatic Rollback System

**Rollback Manager**:
```python
# bem2/deployment/rollback_manager.py
class RollbackManager:
    """
    Automatic rollback system for production deployments.
    Monitors key metrics and triggers rollbacks on violations.
    """
    
    def __init__(self, config: RollbackConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.deployment_history = DeploymentHistory()
        self.rollback_executor = RollbackExecutor()
        
    def monitor_deployment(self, deployment_id: str):
        """Monitor deployment and trigger rollback if needed"""
        start_time = time.time()
        
        while time.time() - start_time < self.config.monitoring_duration:
            metrics = self.metrics_collector.get_current_metrics()
            
            # Check rollback triggers
            if self._should_rollback(metrics):
                logger.error(f"Rollback triggered for deployment {deployment_id}")
                self._execute_rollback(deployment_id)
                return RollbackResult.TRIGGERED
                
            time.sleep(self.config.check_interval)
            
        return RollbackResult.MONITORING_COMPLETE
        
    def _should_rollback(self, metrics: ProductionMetrics) -> bool:
        """Determine if rollback should be triggered"""
        rollback_triggers = []
        
        # Error rate check
        if metrics.error_rate > self.config.max_error_rate:
            rollback_triggers.append(f"Error rate {metrics.error_rate} > {self.config.max_error_rate}")
            
        # Latency check  
        if metrics.p95_latency > self.config.max_latency:
            rollback_triggers.append(f"p95 latency {metrics.p95_latency} > {self.config.max_latency}")
            
        # Safety violations
        if metrics.safety_violations > 0:
            rollback_triggers.append(f"Safety violations detected: {metrics.safety_violations}")
            
        # Model accuracy degradation
        if metrics.accuracy_drop > self.config.max_accuracy_drop:
            rollback_triggers.append(f"Accuracy dropped by {metrics.accuracy_drop}")
            
        if rollback_triggers:
            logger.warning(f"Rollback triggers: {rollback_triggers}")
            return True
            
        return False
        
    def _execute_rollback(self, deployment_id: str):
        """Execute rollback to previous stable version"""
        try:
            # Get previous stable version
            previous_version = self.deployment_history.get_previous_stable()
            
            # Execute rollback
            self.rollback_executor.rollback_to_version(previous_version)
            
            # Update deployment history
            self.deployment_history.mark_rollback(deployment_id, previous_version)
            
            # Send notifications
            self._send_rollback_notification(deployment_id, previous_version)
            
        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            self._send_emergency_notification(deployment_id, str(e))
```

## ⚡ Performance Optimization

### Production Performance Tuning

**GPU Optimization**:
```python
# bem2/optimization/gpu_optimizer.py
class ProductionGPUOptimizer:
    """GPU optimization for production deployment"""
    
    def __init__(self):
        self.cuda_memory_manager = CudaMemoryManager()
        self.kernel_cache = {}
        self.compiled_models = {}
        
    def optimize_for_production(self, model):
        """Apply production optimizations"""
        
        # 1. Model compilation with TorchScript
        compiled_model = torch.jit.script(model)
        compiled_model = torch.jit.optimize_for_inference(compiled_model)
        
        # 2. CUDA kernel optimization
        self._optimize_cuda_kernels()
        
        # 3. Memory pool configuration
        torch.cuda.memory.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()
        
        # 4. Mixed precision setup
        if torch.cuda.get_device_capability()[0] >= 7:  # Tensor cores available
            compiled_model = compiled_model.half()
            
        return compiled_model
        
    def _optimize_cuda_kernels(self):
        """Optimize CUDA kernels for production"""
        # Pre-compile frequently used kernels
        kernel_configs = [
            ('dynamic_rank_kernel', (1024, 768, 64)),
            ('kronecker_kernel', (512, 1024, 128)),
            ('film_modulation_kernel', (2048, 768))
        ]
        
        for kernel_name, shape in kernel_configs:
            if kernel_name not in self.kernel_cache:
                kernel = self._compile_kernel(kernel_name, shape)
                self.kernel_cache[kernel_name] = kernel
                
        # Optimize memory access patterns
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
```

### Batch Processing Optimization

**Efficient Batch Processing**:
```python
# bem2/optimization/batch_processor.py
class ProductionBatchProcessor:
    """Optimized batch processing for production workloads"""
    
    def __init__(self, model, batch_size=32, max_seq_len=512):
        self.model = model
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.token_bucket = TokenBucket()
        
    def process_batch(self, requests):
        """Process batch with optimization"""
        
        # 1. Dynamic batching with padding optimization
        batches = self._create_optimal_batches(requests)
        
        # 2. Parallel processing with GPU streams
        with torch.cuda.stream(self.compute_stream):
            results = []
            for batch in batches:
                # Pre-process on CPU while GPU computes
                preprocessed = self._preprocess_batch(batch)
                
                # GPU computation
                with torch.no_grad():
                    outputs = self.model(preprocessed)
                    
                # Post-process results
                batch_results = self._postprocess_outputs(outputs)
                results.extend(batch_results)
                
        return results
        
    def _create_optimal_batches(self, requests):
        """Create batches optimized for GPU utilization"""
        # Sort by sequence length for efficient padding
        sorted_requests = sorted(requests, key=lambda x: len(x['tokens']))
        
        batches = []
        current_batch = []
        current_max_len = 0
        
        for request in sorted_requests:
            seq_len = len(request['tokens'])
            
            # Check if adding this request would exceed memory
            projected_memory = self._estimate_memory_usage(
                len(current_batch) + 1,
                max(current_max_len, seq_len)
            )
            
            if (projected_memory > self.memory_budget or 
                len(current_batch) >= self.batch_size):
                # Finalize current batch
                if current_batch:
                    batches.append(self._finalize_batch(current_batch))
                current_batch = [request]
                current_max_len = seq_len
            else:
                current_batch.append(request)
                current_max_len = max(current_max_len, seq_len)
                
        # Add final batch
        if current_batch:
            batches.append(self._finalize_batch(current_batch))
            
        return batches
```

## 🐛 Troubleshooting

### Common Production Issues

**Issue 1: Out of Memory Errors**
```python
# Diagnostic script
def diagnose_memory_issue():
    """Diagnose GPU memory issues"""
    print("=== GPU Memory Diagnostic ===")
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Current device: {device}")
        print(f"Total memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved(device) / 1e9:.1f} GB")
        print(f"Free memory: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1e9:.1f} GB")
        
        # Memory summary
        print("\nMemory Summary:")
        print(torch.cuda.memory_summary())
        
    # Recommendations
    print("\n=== Recommendations ===")
    print("1. Reduce batch size in config/production.yml")
    print("2. Enable gradient checkpointing")
    print("3. Use mixed precision training")
    print("4. Clear cache: torch.cuda.empty_cache()")
```

**Solution**:
```bash
# Immediate fix
kubectl patch deployment bem-v13-deployment -p '{"spec":{"template":{"spec":{"containers":[{"name":"bem-v13","resources":{"requests":{"memory":"8Gi"},"limits":{"memory":"16Gi"}}}]}}}}'

# Update config
# In config/production.yml:
# training:
#   batch_size: 8  # Reduced from 16
#   gradient_accumulation_steps: 4
#   use_gradient_checkpointing: true
```

**Issue 2: High Latency**
```python
# Latency diagnostic
def diagnose_latency_issue():
    """Diagnose latency issues"""
    
    # Profile request processing
    profiler = BEMProfiler()
    
    with profiler.profile_request():
        # Simulate typical request
        result = model.process_request(sample_request)
        
    # Analyze bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    
    print("=== Latency Analysis ===")
    for bottleneck in bottlenecks:
        print(f"{bottleneck.component}: {bottleneck.duration:.3f}s ({bottleneck.percentage:.1f}%)")
        
    # Specific checks
    if profiler.gpu_utilization < 70:
        print("WARNING: Low GPU utilization - check batch size")
        
    if profiler.memory_transfer_time > 0.1:
        print("WARNING: High memory transfer time - optimize data loading")
```

### Health Check Endpoints

**Comprehensive Health Checks**:
```python
# bem2/api/health.py
@app.route('/health')
def health_check():
    """Basic health check"""
    return {'status': 'healthy', 'timestamp': time.time()}
    
@app.route('/ready')
def readiness_check():
    """Readiness check with component validation"""
    checks = {}
    overall_status = 'ready'
    
    # Model loading check
    try:
        if hasattr(app.model, 'forward'):
            checks['model'] = 'ready'
        else:
            checks['model'] = 'not_ready'
            overall_status = 'not_ready'
    except Exception as e:
        checks['model'] = f'error: {e}'
        overall_status = 'not_ready'
        
    # GPU availability check
    if torch.cuda.is_available():
        checks['gpu'] = 'available'
    else:
        checks['gpu'] = 'not_available'
        overall_status = 'not_ready'
        
    # Memory check
    if torch.cuda.is_available():
        memory_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        if memory_percent < 0.9:
            checks['memory'] = 'ok'
        else:
            checks['memory'] = 'high_usage'
            overall_status = 'degraded'
            
    return {
        'status': overall_status,
        'checks': checks,
        'timestamp': time.time()
    }
    
@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()
```

## 🚨 Disaster Recovery

### Backup Strategy

**Automated Backup System**:
```python
# bem2/backup/backup_manager.py
class ProductionBackupManager:
    """Production backup and recovery system"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.db_client = DatabaseClient()
        
    def create_full_backup(self):
        """Create complete system backup"""
        backup_id = f"bem-v13-{datetime.now().isoformat()}"
        
        try:
            # 1. Backup model artifacts
            model_backup = self._backup_models(backup_id)
            
            # 2. Backup configuration
            config_backup = self._backup_configuration(backup_id)
            
            # 3. Backup database
            db_backup = self._backup_database(backup_id)
            
            # 4. Create backup manifest
            manifest = {
                'backup_id': backup_id,
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'models': model_backup,
                    'config': config_backup,
                    'database': db_backup
                },
                'version': 'v1.3.0',
                'status': 'complete'
            }
            
            # 5. Store manifest
            self._store_backup_manifest(backup_id, manifest)
            
            return BackupResult(success=True, backup_id=backup_id)
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return BackupResult(success=False, error=str(e))
```

### Recovery Procedures

**Disaster Recovery Playbook**:
```bash
#!/bin/bash
# disaster_recovery.sh

echo "=== BEM v1.3 Disaster Recovery ==="

# 1. Assess damage
echo "Step 1: Assessing system status..."
kubectl get pods -l app=bem-v13
kubectl get nodes

# 2. Restore from backup
echo "Step 2: Restoring from backup..."
BACKUP_ID=$(python3 backup/get_latest_backup.py)
python3 backup/restore_backup.py --backup-id $BACKUP_ID

# 3. Verify restoration
echo "Step 3: Verifying restoration..."
python3 deployment/validate_production.py --quick-check

# 4. Gradual traffic restoration
echo "Step 4: Restoring traffic gradually..."
python3 deployment/canary_controller.py \
  --restore-traffic \
  --gradual \
  --safety-checks-enabled

# 5. Full system validation
echo "Step 5: Full system validation..."
python3 test_bem_v13_comprehensive.py --production-mode

echo "Disaster recovery complete!"
```

This comprehensive deployment guide ensures production-ready deployment of the BEM v1.3 system with enterprise-grade safety, monitoring, and operational capabilities.