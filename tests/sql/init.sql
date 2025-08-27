-- Test database initialization script
-- This script sets up the test database schema and initial data

-- Create test database (if not exists)
SELECT 'CREATE DATABASE bem_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'bem_test');

-- Connect to test database
\c bem_test;

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- For GIN indexes

-- Create schemas
CREATE SCHEMA IF NOT EXISTS bem;
CREATE SCHEMA IF NOT EXISTS experiments;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO bem, experiments, monitoring, public;

-- ============================================================================
-- Core BEM tables
-- ============================================================================

-- Models table
CREATE TABLE IF NOT EXISTS bem.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    config JSONB NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Model checkpoints
CREATE TABLE IF NOT EXISTS bem.model_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES bem.models(id) ON DELETE CASCADE,
    checkpoint_path VARCHAR(500) NOT NULL,
    step INTEGER NOT NULL,
    metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    file_size BIGINT,
    checksum VARCHAR(64)
);

-- Training runs
CREATE TABLE IF NOT EXISTS bem.training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES bem.models(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metrics JSONB DEFAULT '{}'::jsonb,
    logs TEXT,
    error_message TEXT
);

-- Evaluation results
CREATE TABLE IF NOT EXISTS bem.evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES bem.models(id) ON DELETE CASCADE,
    dataset_name VARCHAR(255) NOT NULL,
    metrics JSONB NOT NULL,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    evaluator_config JSONB DEFAULT '{}'::jsonb,
    notes TEXT
);

-- ============================================================================
-- Experiment tracking tables
-- ============================================================================

-- Experiments
CREATE TABLE IF NOT EXISTS experiments.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Experiment runs
CREATE TABLE IF NOT EXISTS experiments.runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments.experiments(id) ON DELETE CASCADE,
    run_name VARCHAR(255) NOT NULL,
    parameters JSONB NOT NULL,
    metrics JSONB DEFAULT '{}'::jsonb,
    artifacts JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    logs TEXT,
    error_message TEXT
);

-- Experiment metrics (time series)
CREATE TABLE IF NOT EXISTS experiments.metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES experiments.runs(id) ON DELETE CASCADE,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    step INTEGER NOT NULL,
    timestamp_utc TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================================
-- Monitoring and logging tables
-- ============================================================================

-- System metrics
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hostname VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    timestamp_utc TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    labels JSONB DEFAULT '{}'::jsonb
);

-- Application logs
CREATE TABLE IF NOT EXISTS monitoring.application_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL,
    logger_name VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    timestamp_utc TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    module VARCHAR(255),
    function_name VARCHAR(255),
    line_number INTEGER,
    extra_data JSONB DEFAULT '{}'::jsonb
);

-- Safety violations
CREATE TABLE IF NOT EXISTS monitoring.safety_violations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES bem.models(id) ON DELETE SET NULL,
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    input_text TEXT,
    output_text TEXT,
    confidence DOUBLE PRECISION,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================================
-- Indexes for performance
-- ============================================================================

-- Models indexes
CREATE INDEX IF NOT EXISTS idx_models_name ON bem.models(name);
CREATE INDEX IF NOT EXISTS idx_models_active ON bem.models(is_active);
CREATE INDEX IF NOT EXISTS idx_models_created_at ON bem.models(created_at);

-- Model checkpoints indexes
CREATE INDEX IF NOT EXISTS idx_checkpoints_model_id ON bem.model_checkpoints(model_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_step ON bem.model_checkpoints(step);

-- Training runs indexes
CREATE INDEX IF NOT EXISTS idx_training_runs_model_id ON bem.training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON bem.training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_started_at ON bem.training_runs(started_at);

-- Evaluations indexes
CREATE INDEX IF NOT EXISTS idx_evaluations_model_id ON bem.evaluations(model_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_dataset ON bem.evaluations(dataset_name);

-- Experiments indexes
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments.experiments(name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments.experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_tags ON experiments.experiments USING GIN(tags);

-- Runs indexes
CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON experiments.runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON experiments.runs(status);

-- Metrics indexes (for time series queries)
CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON experiments.metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON experiments.metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON experiments.metrics(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_metrics_run_name_step ON experiments.metrics(run_id, metric_name, step);

-- Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_hostname ON monitoring.system_metrics(hostname);
CREATE INDEX IF NOT EXISTS idx_system_metrics_type ON monitoring.system_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON monitoring.system_metrics(timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_app_logs_level ON monitoring.application_logs(level);
CREATE INDEX IF NOT EXISTS idx_app_logs_logger ON monitoring.application_logs(logger_name);
CREATE INDEX IF NOT EXISTS idx_app_logs_timestamp ON monitoring.application_logs(timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_safety_violations_type ON monitoring.safety_violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_safety_violations_severity ON monitoring.safety_violations(severity);
CREATE INDEX IF NOT EXISTS idx_safety_violations_detected_at ON monitoring.safety_violations(detected_at);

-- ============================================================================
-- Functions and triggers
-- ============================================================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at columns
CREATE TRIGGER update_models_updated_at
    BEFORE UPDATE ON bem.models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Initial test data
-- ============================================================================

-- Insert test models
INSERT INTO bem.models (name, config, version) VALUES
('test-model-small', '{"hidden_size": 64, "num_layers": 2}', '1.0.0'),
('test-model-medium', '{"hidden_size": 128, "num_layers": 4}', '1.0.0'),
('test-model-large', '{"hidden_size": 256, "num_layers": 8}', '1.0.0')
ON CONFLICT (name) DO NOTHING;

-- Insert test experiments
INSERT INTO experiments.experiments (name, description, config) VALUES
('baseline-experiment', 'Baseline model training', '{"learning_rate": 1e-4, "batch_size": 32}'),
('ablation-study', 'Ablation study for expert routing', '{"learning_rate": 5e-5, "num_experts": 4}'),
('safety-evaluation', 'Safety and alignment testing', '{"safety_threshold": 0.8}')
ON CONFLICT (name) DO NOTHING;

-- Create test user for integration tests
CREATE USER bem_test_user WITH PASSWORD 'test_password';
GRANT USAGE ON SCHEMA bem, experiments, monitoring TO bem_test_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA bem, experiments, monitoring TO bem_test_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA bem, experiments, monitoring TO bem_test_user;

-- ============================================================================
-- Views for common queries
-- ============================================================================

-- Active models with latest metrics
CREATE OR REPLACE VIEW bem.active_models_summary AS
SELECT 
    m.id,
    m.name,
    m.version,
    m.created_at,
    COUNT(c.id) as checkpoint_count,
    MAX(c.step) as latest_step,
    COUNT(e.id) as evaluation_count
FROM bem.models m
LEFT JOIN bem.model_checkpoints c ON m.id = c.model_id
LEFT JOIN bem.evaluations e ON m.id = e.model_id
WHERE m.is_active = true
GROUP BY m.id, m.name, m.version, m.created_at;

-- Recent experiment runs with status
CREATE OR REPLACE VIEW experiments.recent_runs AS
SELECT 
    r.id,
    r.run_name,
    e.name as experiment_name,
    r.status,
    r.started_at,
    r.completed_at,
    CASE 
        WHEN r.completed_at IS NOT NULL AND r.started_at IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (r.completed_at - r.started_at))
        ELSE NULL 
    END as duration_seconds
FROM experiments.runs r
JOIN experiments.experiments e ON r.experiment_id = e.id
ORDER BY r.started_at DESC
LIMIT 100;

-- Safety violation summary
CREATE OR REPLACE VIEW monitoring.safety_summary AS
SELECT 
    violation_type,
    severity,
    COUNT(*) as violation_count,
    AVG(confidence) as avg_confidence,
    MAX(detected_at) as latest_detection
FROM monitoring.safety_violations
WHERE detected_at >= NOW() - INTERVAL '30 days'
GROUP BY violation_type, severity
ORDER BY violation_count DESC;

COMMIT;