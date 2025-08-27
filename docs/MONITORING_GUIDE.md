# BEM Fleet Monitoring Guide

## 📊 Real-Time Monitoring System

The BEM Fleet includes a comprehensive monitoring system that provides real-time visibility into mission progress, system performance, and statistical validation across all 5 parallel research tracks.

## 🎛️ Fleet Dashboard

### Launching the Dashboard

```bash
# Start the monitoring dashboard
python monitoring/fleet_dashboard.py \
  --config configs/fleet_monitoring.yaml \
  --port 8080 \
  --refresh-interval 30

# Access dashboard at http://localhost:8080
```

### Dashboard Components

#### 1. Fleet Overview (Main View)
```
┌─────────────────────────────────────────────────────────────────┐
│                    BEM Fleet Command Center                     │
├─────────────────────────────────────────────────────────────────┤
│ Mission Status Grid:                                            │
│ ┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓ │
│ ┃Mission A  ┃Mission B  ┃Mission C  ┃Mission D  ┃Mission E  ┃ │
│ ┃Router     ┃Online     ┃Safety     ┃SEP        ┃Memory     ┃ │
│ ┃🟢 RUNNING ┃🟡 ADAPT   ┃✅ COMPLETE┃🔵 TRAIN   ┃⏸️ QUEUE   ┃ │
│ ┃+1.8% EM/F1┃892 prompts┃-32% viol  ┃Epoch 23/50┃Pending    ┃ │
│ ┗━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━━┛ │
│                                                                 │
│ Resource Utilization:                                           │
│ GPU 0: [████████████████████████████████████░░░░] 87% (H100)   │
│ GPU 1: [██████████████████████████░░░░░░░░░░░░░░░] 62% (4090)   │
│ RAM:   [██████████████████████████████░░░░░░░░░░] 73% 128GB     │
│ Disk:  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 34% 2TB       │
│                                                                 │
│ System Health: 🟢 HEALTHY │ Uptime: 2d 14h 23m │ Temp: 67°C   │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Mission Details Panel
Individual mission monitoring with real-time metrics:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mission A: Agentic Planner                 │
├─────────────────────────────────────────────────────────────────┤
│ Training Progress:                                              │
│ ┌─ Loss Curve ─────────────────────────────────────────────────┐│
│ │ 3.2│                                                         ││
│ │    │\                                                        ││
│ │ 2.8│ \                                                       ││
│ │    │  \_                                                     ││
│ │ 2.4│    \___                                                 ││
│ │    │        \____                                            ││
│ │ 2.0│             \________                                   ││
│ │    │                      \____________                      ││
│ │ 1.6│                                   \__________           ││
│ │    │0        5k       10k      15k      20k      25k       ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ Current Metrics:                                               │
│ • EM/F1 Score: 87.3% (+1.8% vs baseline)                     │
│ • Plan Length: 2.1 (target: ≤3.0)                            │
│ • KV Hit Ratio: 1.12x baseline                                │
│ • Latency P95: 145ms (+12% overhead)                          │
│                                                                 │
│ Statistical Status:                                             │
│ • Bootstrap CI: [0.85, 0.89] (95% confident)                  │
│ • Effect Size: d = 0.67 (medium-large)                        │
│ • P-value: < 0.001 (highly significant)                       │
│ • Gates Status: ✅ PASSING (3/4 gates met)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Dashboard Features

#### Real-Time Metrics Tracking
```python
class RealTimeMetrics:
    """Real-time metrics collection and display"""
    
    def __init__(self):
        self.metrics = {
            'mission_progress': {},
            'resource_utilization': {},
            'statistical_significance': {},
            'error_rates': {},
            'integration_health': {}
        }
        
    def collect_mission_metrics(self, mission):
        """Collect comprehensive mission metrics"""
        return {
            'training_progress': {
                'current_epoch': mission.current_epoch,
                'total_epochs': mission.total_epochs,
                'loss': mission.current_loss,
                'learning_rate': mission.current_lr,
                'samples_processed': mission.samples_processed
            },
            'performance_metrics': {
                'primary_metric': mission.primary_metric_value,
                'secondary_metrics': mission.secondary_metrics,
                'baseline_comparison': mission.vs_baseline,
                'trend': mission.performance_trend
            },
            'statistical_metrics': {
                'confidence_interval': mission.confidence_interval,
                'effect_size': mission.effect_size,
                'p_value': mission.p_value,
                'significance': mission.is_significant
            },
            'resource_metrics': {
                'gpu_utilization': mission.gpu_usage,
                'memory_usage': mission.memory_usage,
                'throughput': mission.samples_per_second,
                'eta': mission.estimated_completion_time
            }
        }
```

## 📈 Performance Monitoring

### Key Performance Indicators (KPIs)

#### Mission-Specific KPIs
```yaml
mission_kpis:
  mission_a:
    primary: "em_f1_improvement"
    target: "≥1.5%"
    current: "+1.8%"
    trend: "improving"
    
  mission_b:
    primary: "correction_time"
    target: "≤1000 prompts"
    current: "892 prompts"
    trend: "stable"
    
  mission_c:
    primary: "violation_reduction"
    target: "≥30%"
    current: "32.1%"
    trend: "stable"
    
  mission_d:
    primary: "ood_improvement"
    target: "measurable"
    current: "training"
    trend: "unknown"
    
  mission_e:
    primary: "long_context_performance"
    target: "superior at 128k+"
    current: "queued"
    trend: "unknown"
```

#### System-Wide KPIs
```yaml
system_kpis:
  overall_health: "healthy"
  mission_success_rate: "60%" # 3/5 missions successful so far
  resource_efficiency: "87%" # GPU utilization
  integration_stability: "stable"
  statistical_power: "adequate"
  estimated_completion: "12 days remaining"
```

### Performance Visualization

#### Training Curves Monitoring
```python
class TrainingCurveMonitor:
    """Real-time training curve visualization"""
    
    def __init__(self, missions):
        self.missions = missions
        self.curve_data = {mission: [] for mission in missions}
        
    def update_curves(self):
        """Update training curves for all missions"""
        for mission_name, mission in self.missions.items():
            if mission.is_training():
                metrics = mission.get_latest_metrics()
                self.curve_data[mission_name].append({
                    'step': metrics['step'],
                    'loss': metrics['loss'],
                    'primary_metric': metrics['primary_metric'],
                    'timestamp': metrics['timestamp']
                })
                
    def plot_curves(self):
        """Generate real-time training curve plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (mission_name, data) in enumerate(self.curve_data.items()):
            if not data:
                continue
                
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            steps = [d['step'] for d in data]
            losses = [d['loss'] for d in data]
            
            ax.plot(steps, losses, label='Training Loss')
            ax.set_title(f'{mission_name.title()} Progress')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.legend()
            
        plt.tight_layout()
        return fig
```

## 🚨 Alert System

### Alert Configuration
```yaml
alert_config:
  # Performance Alerts
  performance_alerts:
    gpu_utilization_low:
      threshold: 50%
      duration: 300s  # 5 minutes
      severity: warning
      
    memory_usage_high:
      threshold: 90%
      duration: 60s
      severity: critical
      
    training_loss_plateau:
      threshold: 1000  # steps without improvement
      severity: warning
      
    statistical_significance_lost:
      threshold: 0.05  # p-value rises above
      severity: high
      
  # System Health Alerts  
  system_alerts:
    mission_failure:
      condition: "mission_status == 'failed'"
      severity: critical
      immediate: true
      
    integration_conflict:
      condition: "integration_errors > 0"
      severity: high
      
    resource_conflict:
      condition: "resource_conflicts_detected"
      severity: medium
      
  # Security Alerts
  security_alerts:
    safety_violation:
      condition: "safety_violations_detected"
      severity: critical
      immediate: true
      
    unauthorized_access:
      condition: "auth_failures > 5"
      severity: critical
      immediate: true
```

### Alert Delivery Systems
```python
class AlertManager:
    """Manage and deliver alerts across multiple channels"""
    
    def __init__(self, config):
        self.config = config
        self.channels = {
            'email': EmailNotifier(config['email']),
            'slack': SlackNotifier(config['slack']),
            'sms': SMSNotifier(config['sms']),
            'webhook': WebhookNotifier(config['webhook'])
        }
        
    def send_alert(self, alert):
        """Send alert through appropriate channels based on severity"""
        channels = self._get_channels_for_severity(alert.severity)
        
        for channel_name in channels:
            try:
                channel = self.channels[channel_name]
                channel.send_alert(alert)
                logger.info(f"Alert sent via {channel_name}: {alert.title}")
            except Exception as e:
                logger.error(f"Failed to send alert via {channel_name}: {e}")
                
    def _get_channels_for_severity(self, severity):
        """Determine which channels to use based on alert severity"""
        severity_channels = {
            'info': ['webhook'],
            'warning': ['slack', 'webhook'],
            'high': ['email', 'slack', 'webhook'],
            'critical': ['email', 'slack', 'sms', 'webhook']
        }
        return severity_channels.get(severity, ['webhook'])
```

## 📊 Statistical Monitoring

### Statistical Significance Tracking
```python
class StatisticalSignificanceMonitor:
    """Monitor statistical significance in real-time"""
    
    def __init__(self, missions, alpha=0.05):
        self.missions = missions
        self.alpha = alpha
        self.significance_history = {}
        
    def check_significance(self, mission_name):
        """Check current statistical significance for a mission"""
        mission = self.missions[mission_name]
        
        if not mission.has_sufficient_data():
            return SignificanceStatus(
                mission=mission_name,
                status='insufficient_data',
                p_value=None,
                confidence_interval=None,
                power=None
            )
            
        # Run statistical test
        treatment_data = mission.get_treatment_results()
        baseline_data = mission.get_baseline_results()
        
        # BCa bootstrap test
        bootstrap_result = mission.statistical_validator.bca_bootstrap(
            treatment_data, baseline_data
        )
        
        # Power analysis
        power = mission.statistical_validator.calculate_power(
            treatment_data, baseline_data
        )
        
        status = SignificanceStatus(
            mission=mission_name,
            status='significant' if bootstrap_result.p_value < self.alpha else 'not_significant',
            p_value=bootstrap_result.p_value,
            confidence_interval=bootstrap_result.confidence_interval,
            effect_size=bootstrap_result.effect_size,
            power=power,
            timestamp=datetime.utcnow()
        )
        
        # Store in history
        if mission_name not in self.significance_history:
            self.significance_history[mission_name] = []
        self.significance_history[mission_name].append(status)
        
        return status
        
    def generate_significance_report(self):
        """Generate comprehensive significance report"""
        report = StatisticalSignificanceReport()
        
        for mission_name in self.missions:
            current_status = self.check_significance(mission_name)
            history = self.significance_history.get(mission_name, [])
            
            report.add_mission_status(
                mission_name,
                current_status,
                trend=self._calculate_significance_trend(history),
                stability=self._calculate_stability(history)
            )
            
        return report
```

### Progress Tracking Dashboard
```
┌─────────────────────────────────────────────────────────────────┐
│                Statistical Progress Overview                    │
├─────────────────────────────────────────────────────────────────┤
│ Mission A: ████████████████████████████████░░░░ 87% Complete    │
│           p=0.001 ✅ | d=0.67 ✅ | CI=[0.85,0.89] ✅           │
│                                                                 │
│ Mission B: ███████████████████████░░░░░░░░░░░░░░ 65% Complete    │
│           p=0.023 ✅ | d=0.45 ⚠️  | CI=[0.12,0.89] ✅           │
│                                                                 │
│ Mission C: ████████████████████████████████████ 100% Complete   │
│           p<0.001 ✅ | d=0.83 ✅ | CI=[0.28,0.36] ✅           │
│                                                                 │
│ Mission D: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 35% Complete    │
│           p=N/A ⏸️ | d=N/A ⏸️ | Training in progress...        │
│                                                                 │
│ Mission E: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0% Complete     │
│           p=N/A ⏸️ | d=N/A ⏸️ | Queued                         │
│                                                                 │
│ Overall FDR Status: 🟡 PARTIAL (3/5 missions significant)       │
│ Expected Completion: 12.3 days                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔍 Log Monitoring

### Centralized Log Aggregation
```python
class LogMonitor:
    """Centralized log monitoring and analysis"""
    
    def __init__(self, log_sources):
        self.log_sources = log_sources
        self.log_parsers = {
            'training': TrainingLogParser(),
            'evaluation': EvaluationLogParser(),
            'integration': IntegrationLogParser(),
            'system': SystemLogParser(),
            'security': SecurityLogParser()
        }
        
    def monitor_logs(self):
        """Real-time log monitoring across all sources"""
        for source_name, source_config in self.log_sources.items():
            log_type = source_config['type']
            parser = self.log_parsers[log_type]
            
            # Tail log file
            for line in self._tail_log_file(source_config['path']):
                parsed_event = parser.parse_line(line)
                
                if parsed_event:
                    self._handle_log_event(source_name, parsed_event)
                    
    def _handle_log_event(self, source, event):
        """Handle parsed log events"""
        # Check for error patterns
        if event.level == 'ERROR':
            self._handle_error_event(source, event)
            
        # Check for performance anomalies
        if event.type == 'performance' and event.is_anomalous():
            self._handle_performance_anomaly(source, event)
            
        # Check for security events
        if event.type == 'security':
            self._handle_security_event(source, event)
            
        # Store for analysis
        self._store_log_event(source, event)
```

### Log Analysis Dashboard
```bash
# Real-time log analysis
python scripts/log_analysis_dashboard.py \
  --sources all \
  --real-time \
  --alert-on-errors \
  --port 8081
```

## 🎛️ Custom Monitoring Setup

### Creating Custom Metrics
```python
class CustomMetricCollector:
    """Create and collect custom metrics"""
    
    def __init__(self):
        self.custom_metrics = {}
        
    def register_metric(self, name, collector_func, frequency=30):
        """Register a custom metric with collection frequency"""
        self.custom_metrics[name] = {
            'collector': collector_func,
            'frequency': frequency,
            'last_collection': 0,
            'values': []
        }
        
    def collect_all_metrics(self):
        """Collect all registered custom metrics"""
        current_time = time.time()
        
        for metric_name, config in self.custom_metrics.items():
            if current_time - config['last_collection'] >= config['frequency']:
                try:
                    value = config['collector']()
                    config['values'].append({
                        'timestamp': current_time,
                        'value': value
                    })
                    config['last_collection'] = current_time
                except Exception as e:
                    logger.error(f"Failed to collect metric {metric_name}: {e}")

# Example custom metrics
def mission_synchronization_score():
    """Custom metric: How well synchronized are the missions"""
    # Implementation here
    return 0.85

def integration_efficiency():
    """Custom metric: Efficiency of cross-mission integration"""
    # Implementation here  
    return 0.92

# Register custom metrics
collector = CustomMetricCollector()
collector.register_metric('mission_sync', mission_synchronization_score, frequency=60)
collector.register_metric('integration_efficiency', integration_efficiency, frequency=120)
```

### Monitoring Configuration
```yaml
# configs/monitoring_config.yaml
monitoring:
  dashboard:
    port: 8080
    refresh_interval: 30  # seconds
    auto_refresh: true
    
  metrics:
    collection_interval: 15  # seconds
    retention_period: "30d"
    aggregation_windows: ["1m", "5m", "1h", "1d"]
    
  alerts:
    enabled: true
    channels: ["email", "slack"]
    severity_thresholds:
      warning: 0.7
      critical: 0.9
      
  logging:
    level: "INFO"
    rotation: "daily"
    retention: "7d"
    
  performance:
    sampling_rate: 0.1  # 10% of requests
    profiling_enabled: false
    memory_tracking: true
```

This comprehensive monitoring guide provides all the tools and information needed to effectively monitor the BEM Fleet system, from real-time dashboards to custom metrics and alert management.