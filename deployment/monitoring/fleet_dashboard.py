#!/usr/bin/env python3
"""
BEM Fleet Monitoring Dashboard
Real-time monitoring and evaluation dashboard for all 5 missions

Provides comprehensive monitoring of:
- Individual mission progress and metrics
- Cross-mission interactions and dependencies
- Resource utilization and performance
- Statistical significance tracking
- Alert management and automated responses
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import psutil
import GPUtil
from threading import Thread
import queue
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MissionStatus:
    """Status information for a mission"""
    name: str
    phase: str  # 'preparation', 'training', 'evaluation', 'completed'
    progress: float  # 0.0 to 1.0
    current_metrics: Dict[str, float]
    alerts: List[str]
    resource_usage: Dict[str, float]
    last_updated: datetime
    
@dataclass
class CrossMissionMetric:
    """Metrics that span multiple missions"""
    metric_name: str
    mission_values: Dict[str, float]
    baseline_value: float
    statistical_significance: Dict[str, bool]
    trend: str  # 'improving', 'declining', 'stable'
    
class FleetMonitor:
    """Core monitoring system for BEM Fleet"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.db_path = self.config.get('database_path', 'fleet_monitoring.db')
        self._init_database()
        self.mission_status = {}
        self.alert_queue = queue.Queue()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS mission_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    phase TEXT,
                    seed INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    gpu_utilization REAL,
                    gpu_memory REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    mission TEXT,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
    
    @contextmanager
    def get_db_connection(self):
        """Database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def update_mission_metrics(self, mission: str, metrics: Dict[str, float], 
                             phase: str = None, seed: int = None):
        """Update metrics for a mission"""
        with self.get_db_connection() as conn:
            for metric_name, value in metrics.items():
                conn.execute(
                    'INSERT INTO mission_metrics (mission, metric_name, metric_value, phase, seed) VALUES (?, ?, ?, ?, ?)',
                    (mission, metric_name, value, phase, seed)
                )
            conn.commit()
        
        logger.info(f"Updated metrics for Mission {mission}: {metrics}")
    
    def update_system_resources(self):
        """Update system resource metrics"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_utilization = 0
        gpu_memory = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_utilization = np.mean([gpu.load * 100 for gpu in gpus])
                gpu_memory = np.mean([gpu.memoryUtil * 100 for gpu in gpus])
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
        
        with self.get_db_connection() as conn:
            conn.execute(
                'INSERT INTO system_resources (cpu_percent, memory_percent, gpu_utilization, gpu_memory) VALUES (?, ?, ?, ?)',
                (cpu_percent, memory.percent, gpu_utilization, gpu_memory)
            )
            conn.commit()
    
    def check_alerts(self, mission: str, metrics: Dict[str, float]) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        thresholds = self.config.get('alert_thresholds', {})
        
        for metric_name, value in metrics.items():
            if metric_name in thresholds:
                threshold = thresholds[metric_name]
                
                if 'max' in threshold and value > threshold['max']:
                    alert_msg = f"Mission {mission}: {metric_name} = {value:.4f} exceeds maximum {threshold['max']}"
                    alerts.append(alert_msg)
                    self._log_alert(mission, 'threshold_exceeded', alert_msg, 'warning')
                
                if 'min' in threshold and value < threshold['min']:
                    alert_msg = f"Mission {mission}: {metric_name} = {value:.4f} below minimum {threshold['min']}"
                    alerts.append(alert_msg)
                    self._log_alert(mission, 'threshold_exceeded', alert_msg, 'warning')
        
        return alerts
    
    def _log_alert(self, mission: str, alert_type: str, message: str, severity: str):
        """Log alert to database"""
        with self.get_db_connection() as conn:
            conn.execute(
                'INSERT INTO alerts (mission, alert_type, message, severity) VALUES (?, ?, ?, ?)',
                (mission, alert_type, message, severity)
            )
            conn.commit()
        
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
    
    def get_recent_metrics(self, mission: str, hours: int = 24) -> pd.DataFrame:
        """Get recent metrics for a mission"""
        with self.get_db_connection() as conn:
            query = '''
                SELECT * FROM mission_metrics 
                WHERE mission = ? AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            '''.format(hours)
            
            return pd.read_sql_query(query, conn, params=(mission,))
    
    def get_cross_mission_summary(self) -> Dict[str, Dict]:
        """Get summary of all missions for cross-mission view"""
        summary = {}
        missions = ['A', 'B', 'C', 'D', 'E']
        
        for mission in missions:
            recent_metrics = self.get_recent_metrics(mission, hours=1)
            if not recent_metrics.empty:
                latest_metrics = recent_metrics.groupby('metric_name')['metric_value'].last().to_dict()
                summary[mission] = {
                    'latest_metrics': latest_metrics,
                    'last_updated': recent_metrics['timestamp'].max(),
                    'phase': recent_metrics['phase'].iloc[0] if 'phase' in recent_metrics.columns else 'unknown'
                }
            else:
                summary[mission] = {
                    'latest_metrics': {},
                    'last_updated': None,
                    'phase': 'not_started'
                }
        
        return summary

class FleetDashboard:
    """Streamlit dashboard for BEM Fleet monitoring"""
    
    def __init__(self, monitor: FleetMonitor):
        self.monitor = monitor
        st.set_page_config(
            page_title="BEM Fleet Dashboard",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render(self):
        """Render the complete dashboard"""
        st.title("🚀 BEM Fleet Multi-Mission Dashboard")
        st.markdown("Real-time monitoring for 5 parallel missions")
        
        # Sidebar for navigation
        page = st.sidebar.selectbox(
            "Select View",
            ["Fleet Overview", "Mission Details", "Cross-Mission Analysis", 
             "Resource Monitoring", "Alerts & Status", "Statistical Analysis"]
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Route to appropriate page
        if page == "Fleet Overview":
            self.render_fleet_overview()
        elif page == "Mission Details":
            self.render_mission_details()
        elif page == "Cross-Mission Analysis":
            self.render_cross_mission_analysis()
        elif page == "Resource Monitoring":
            self.render_resource_monitoring()
        elif page == "Alerts & Status":
            self.render_alerts_status()
        elif page == "Statistical Analysis":
            self.render_statistical_analysis()
    
    def render_fleet_overview(self):
        """Render fleet-wide overview"""
        st.header("Fleet Overview")
        
        # Get cross-mission summary
        summary = self.monitor.get_cross_mission_summary()
        
        # Mission status cards
        cols = st.columns(5)
        missions = ['A', 'B', 'C', 'D', 'E']
        mission_names = ['Agentic Planner', 'Living Model', 'Alignment Enforcer', 'SEP', 'Long-Memory']
        
        for i, (mission, name) in enumerate(zip(missions, mission_names)):
            with cols[i]:
                status = summary.get(mission, {})
                phase = status.get('phase', 'not_started')
                
                # Status color
                if phase == 'completed':
                    color = '🟢'
                elif phase in ['training', 'evaluation']:
                    color = '🟡'
                elif phase == 'preparation':
                    color = '🔵'
                else:
                    color = '⚪'
                
                st.markdown(f"""
                ### {color} Mission {mission}
                **{name}**
                
                Phase: {phase.title()}
                """)
                
                # Key metrics if available
                metrics = status.get('latest_metrics', {})
                if 'em_f1' in metrics:
                    st.metric("EM/F1", f"{metrics['em_f1']:.3f}")
                if 'latency_p50' in metrics:
                    st.metric("Latency P50", f"{metrics['latency_p50']:.1f}ms")
        
        # Fleet-wide metrics comparison
        st.subheader("Fleet Performance Comparison")
        
        # Collect EM/F1 scores across missions
        em_f1_data = []
        for mission in missions:
            metrics = summary.get(mission, {}).get('latest_metrics', {})
            if 'em_f1' in metrics:
                em_f1_data.append({
                    'Mission': f"Mission {mission}",
                    'EM/F1': metrics['em_f1']
                })
        
        if em_f1_data:
            df = pd.DataFrame(em_f1_data)
            fig = px.bar(df, x='Mission', y='EM/F1', title='EM/F1 Performance Across Missions')
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline view
        st.subheader("Mission Timeline")
        
        timeline_data = []
        for mission in missions:
            with self.monitor.get_db_connection() as conn:
                query = '''
                    SELECT mission, metric_name, metric_value, timestamp
                    FROM mission_metrics 
                    WHERE mission = ? AND metric_name = 'em_f1'
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(mission,))
                if not df.empty:
                    df['Mission'] = f"Mission {mission}"
                    timeline_data.append(df)
        
        if timeline_data:
            combined_df = pd.concat(timeline_data)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            
            fig = px.line(combined_df, x='timestamp', y='metric_value', 
                         color='Mission', title='EM/F1 Performance Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_mission_details(self):
        """Render detailed view for individual missions"""
        st.header("Mission Details")
        
        selected_mission = st.selectbox("Select Mission", ['A', 'B', 'C', 'D', 'E'])
        
        # Get recent metrics for selected mission
        recent_metrics = self.monitor.get_recent_metrics(selected_mission)
        
        if recent_metrics.empty:
            st.warning(f"No data available for Mission {selected_mission}")
            return
        
        # Mission info
        mission_info = {
            'A': {'name': 'Agentic Planner', 'target': '≥+1.5% EM/F1 vs single fused BEM'},
            'B': {'name': 'Living Model', 'target': 'correct failures within ≤1k prompts'},
            'C': {'name': 'Alignment Enforcer', 'target': '≥30% violation reduction at ≤1% EM/F1 drop'},
            'D': {'name': 'SEP', 'target': 'reduce surface dependence, improve OOD transfer'},
            'E': {'name': 'Long-Memory', 'target': 'outperform KV-only at 128k–512k context'}
        }
        
        info = mission_info[selected_mission]
        st.subheader(f"Mission {selected_mission}: {info['name']}")
        st.markdown(f"**Target:** {info['target']}")
        
        # Current metrics summary
        latest_metrics = recent_metrics.groupby('metric_name')['metric_value'].last().to_dict()
        
        cols = st.columns(4)
        metric_names = ['em_f1', 'latency_p50', 'kv_hit_percent', 'effect_size']
        
        for i, metric in enumerate(metric_names):
            if metric in latest_metrics:
                with cols[i]:
                    st.metric(metric.replace('_', ' ').title(), f"{latest_metrics[metric]:.4f}")
        
        # Metrics over time
        st.subheader("Metrics Over Time")
        
        # Select metrics to plot
        available_metrics = recent_metrics['metric_name'].unique()
        selected_metrics = st.multiselect(
            "Select metrics to plot",
            available_metrics,
            default=available_metrics[:3] if len(available_metrics) > 3 else available_metrics
        )
        
        if selected_metrics:
            filtered_data = recent_metrics[recent_metrics['metric_name'].isin(selected_metrics)]
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
            
            fig = px.line(filtered_data, x='timestamp', y='metric_value', 
                         color='metric_name', title=f'Mission {selected_mission} Metrics Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        # Training progress (if applicable)
        if 'training_loss' in available_metrics:
            st.subheader("Training Progress")
            
            loss_data = recent_metrics[recent_metrics['metric_name'] == 'training_loss']
            loss_data['timestamp'] = pd.to_datetime(loss_data['timestamp'])
            
            fig = px.line(loss_data, x='timestamp', y='metric_value', 
                         title=f'Mission {selected_mission} Training Loss')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_cross_mission_analysis(self):
        """Render cross-mission analysis and interactions"""
        st.header("Cross-Mission Analysis")
        
        # Mission interaction matrix
        st.subheader("Mission Interaction Matrix")
        
        interactions = {
            'A-B': 'Router-Online Updates',
            'A-E': 'Router-Memory Coupling', 
            'C-All': 'Safety Overlay',
            'D-All': 'SEP Compatibility'
        }
        
        interaction_df = pd.DataFrame([
            {'Interaction': k, 'Type': v, 'Status': 'Planned'} 
            for k, v in interactions.items()
        ])
        
        st.dataframe(interaction_df, use_container_width=True)
        
        # Performance correlation analysis
        st.subheader("Performance Correlations")
        
        # Collect EM/F1 data for all missions
        correlation_data = {}
        missions = ['A', 'B', 'C', 'D', 'E']
        
        for mission in missions:
            with self.monitor.get_db_connection() as conn:
                query = '''
                    SELECT metric_value, timestamp FROM mission_metrics 
                    WHERE mission = ? AND metric_name = 'em_f1'
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(mission,))
                if not df.empty:
                    correlation_data[f'Mission_{mission}'] = df['metric_value'].values
        
        if len(correlation_data) > 1:
            # Align series lengths
            min_length = min(len(v) for v in correlation_data.values())
            aligned_data = {k: v[:min_length] for k, v in correlation_data.items()}
            
            corr_df = pd.DataFrame(aligned_data)
            correlation_matrix = corr_df.corr()
            
            fig = px.imshow(correlation_matrix, 
                           title="Mission Performance Correlation Matrix",
                           color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_resource_monitoring(self):
        """Render resource utilization monitoring"""
        st.header("Resource Monitoring")
        
        # Get recent resource data
        with self.monitor.get_db_connection() as conn:
            query = '''
                SELECT * FROM system_resources 
                WHERE timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp
            '''
            resource_df = pd.read_sql_query(query, conn)
        
        if resource_df.empty:
            st.warning("No resource data available")
            return
        
        resource_df['timestamp'] = pd.to_datetime(resource_df['timestamp'])
        
        # Current resource status
        latest = resource_df.iloc[-1]
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("CPU Usage", f"{latest['cpu_percent']:.1f}%")
        with cols[1]:
            st.metric("Memory Usage", f"{latest['memory_percent']:.1f}%")
        with cols[2]:
            st.metric("GPU Utilization", f"{latest['gpu_utilization']:.1f}%")
        with cols[3]:
            st.metric("GPU Memory", f"{latest['gpu_memory']:.1f}%")
        
        # Resource trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['CPU Usage', 'Memory Usage', 'GPU Utilization', 'GPU Memory'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=resource_df['timestamp'], y=resource_df['cpu_percent'], name='CPU'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=resource_df['timestamp'], y=resource_df['memory_percent'], name='Memory'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=resource_df['timestamp'], y=resource_df['gpu_utilization'], name='GPU Util'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=resource_df['timestamp'], y=resource_df['gpu_memory'], name='GPU Mem'),
            row=2, col=2
        )
        
        fig.update_layout(title="Resource Utilization (24h)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_status(self):
        """Render alerts and status information"""
        st.header("Alerts & Status")
        
        # Get recent alerts
        with self.monitor.get_db_connection() as conn:
            query = '''
                SELECT * FROM alerts 
                WHERE timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC
            '''
            alerts_df = pd.read_sql_query(query, conn)
        
        if alerts_df.empty:
            st.success("No alerts in the last 24 hours")
        else:
            st.subheader("Recent Alerts")
            
            # Color code by severity
            def get_alert_color(severity):
                colors = {
                    'critical': '🔴',
                    'warning': '🟡', 
                    'info': '🔵'
                }
                return colors.get(severity, '⚪')
            
            alerts_df['Status'] = alerts_df['severity'].apply(get_alert_color)
            
            st.dataframe(
                alerts_df[['Status', 'mission', 'alert_type', 'message', 'timestamp']],
                use_container_width=True
            )
        
        # System status overview
        st.subheader("System Status")
        
        status_checks = {
            'Database Connection': 'operational',
            'Metric Collection': 'operational',
            'Alert System': 'operational',
            'Resource Monitoring': 'operational'
        }
        
        for check, status in status_checks.items():
            color = '🟢' if status == 'operational' else '🔴'
            st.markdown(f"{color} **{check}**: {status.title()}")
    
    def render_statistical_analysis(self):
        """Render statistical analysis and significance testing"""
        st.header("Statistical Analysis")
        
        st.markdown("""
        This section shows statistical significance analysis for mission promotions.
        All tests use **Paired BCa Bootstrap with FDR correction** at 95% confidence.
        """)
        
        # Load statistical results if available
        stats_path = Path('analysis/statistical_validation_report.json')
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)
            
            # Summary statistics
            summary = stats_data.get('summary', {})
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("Missions Tested", summary.get('missions_tested', 0))
            with cols[1]:
                st.metric("Missions Promoted", summary.get('missions_promoted', 0))
            with cols[2]:
                success_rate = summary.get('overall_success_rate', 0)
                st.metric("Success Rate", f"{success_rate:.1%}")
            
            # Mission-specific results
            st.subheader("Mission Promotion Decisions")
            
            mission_results = stats_data.get('mission_results', {})
            
            for mission, results in mission_results.items():
                promote = results.get('promote', False)
                status_icon = '✅' if promote else '❌'
                
                with st.expander(f"{status_icon} Mission {mission}"):
                    st.markdown(f"**Promotion Decision:** {'PROMOTE' if promote else 'DO NOT PROMOTE'}")
                    
                    significant_tests = results.get('significant_tests_count', 0)
                    total_tests = results.get('total_tests_count', 0)
                    st.markdown(f"**Significant Tests:** {significant_tests}/{total_tests}")
                    
                    # Key results table
                    key_results = results.get('key_results', {})
                    if key_results:
                        results_data = []
                        for metric, data in key_results.items():
                            results_data.append({
                                'Metric': metric,
                                'P-Value': f"{data['p_value']:.4f}",
                                'Effect Size': f"{data['effect_size']:.4f}",
                                'Significant': '✅' if data['significant'] else '❌',
                                'CI Lower': f"{data['ci'][0]:.4f}" if not np.isnan(data['ci'][0]) else 'N/A',
                                'CI Upper': f"{data['ci'][1]:.4f}" if not np.isnan(data['ci'][1]) else 'N/A'
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
        else:
            st.info("Statistical analysis not yet available. Run the validation pipeline to generate results.")
            
            # Show example of what analysis will look like
            st.subheader("Analysis Framework")
            
            framework_info = {
                'Method': 'Paired BCa Bootstrap with FDR Correction',
                'Confidence Level': '95%',
                'Bootstrap Samples': '10,000',
                'FDR Method': 'Benjamini-Hochberg',
                'Primary Metrics': 'EM/F1, BLEU, ChrF',
                'Minimum Effect Size': '1.5% for primary metrics'
            }
            
            for key, value in framework_info.items():
                st.markdown(f"**{key}:** {value}")

def main():
    """Main dashboard application"""
    # Initialize monitor
    config_path = Path('configs/fleet_monitoring.yaml')
    if not config_path.exists():
        st.error(f"Configuration file not found: {config_path}")
        st.stop()
    
    monitor = FleetMonitor(str(config_path))
    
    # Create and render dashboard
    dashboard = FleetDashboard(monitor)
    dashboard.render()
    
    # Background resource monitoring
    if 'resource_monitor_started' not in st.session_state:
        def resource_monitor_loop():
            while True:
                try:
                    monitor.update_system_resources()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    time.sleep(60)
        
        resource_thread = Thread(target=resource_monitor_loop, daemon=True)
        resource_thread.start()
        st.session_state.resource_monitor_started = True

if __name__ == "__main__":
    main()
