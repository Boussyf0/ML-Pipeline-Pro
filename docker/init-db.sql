-- Initialize MLOps database schema

-- Create database if it doesn't exist (handled by docker-compose)

-- Create schema for MLflow
CREATE SCHEMA IF NOT EXISTS mlflow;

-- Create schema for application data
CREATE SCHEMA IF NOT EXISTS app_data;

-- Create schema for monitoring
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create table for model performance metrics
CREATE TABLE IF NOT EXISTS monitoring.model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    environment VARCHAR(50) DEFAULT 'production'
);

-- Create table for data drift monitoring
CREATE TABLE IF NOT EXISTS monitoring.data_drift (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(255) NOT NULL,
    drift_score FLOAT NOT NULL,
    drift_detected BOOLEAN DEFAULT FALSE,
    reference_period_start TIMESTAMP NOT NULL,
    reference_period_end TIMESTAMP NOT NULL,
    current_period_start TIMESTAMP NOT NULL,
    current_period_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for A/B test results
CREATE TABLE IF NOT EXISTS monitoring.ab_test_results (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL,
    model_a_name VARCHAR(255) NOT NULL,
    model_b_name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    model_a_value FLOAT NOT NULL,
    model_b_value FLOAT NOT NULL,
    p_value FLOAT,
    is_significant BOOLEAN DEFAULT FALSE,
    winner VARCHAR(10), -- 'A', 'B', or 'tie'
    sample_size_a INTEGER NOT NULL,
    sample_size_b INTEGER NOT NULL,
    test_duration_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for prediction logs
CREATE TABLE IF NOT EXISTS app_data.prediction_logs (
    id SERIAL PRIMARY KEY,
    request_id UUID DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    input_features JSONB NOT NULL,
    prediction FLOAT NOT NULL,
    prediction_proba FLOAT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    session_id VARCHAR(255)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_model_metrics_name_version ON monitoring.model_metrics(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_model_metrics_created_at ON monitoring.model_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_data_drift_feature ON monitoring.data_drift(feature_name);
CREATE INDEX IF NOT EXISTS idx_data_drift_created_at ON monitoring.data_drift(created_at);
CREATE INDEX IF NOT EXISTS idx_ab_test_experiment ON monitoring.ab_test_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model ON app_data.prediction_logs(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON app_data.prediction_logs(created_at);

-- Create views for common queries
CREATE OR REPLACE VIEW monitoring.latest_model_metrics AS
SELECT DISTINCT ON (model_name, metric_name) 
    model_name, 
    model_version, 
    metric_name, 
    metric_value, 
    created_at
FROM monitoring.model_metrics 
ORDER BY model_name, metric_name, created_at DESC;

CREATE OR REPLACE VIEW monitoring.daily_prediction_counts AS
SELECT 
    DATE(created_at) as prediction_date,
    model_name,
    model_version,
    COUNT(*) as prediction_count,
    AVG(response_time_ms) as avg_response_time_ms
FROM app_data.prediction_logs
GROUP BY DATE(created_at), model_name, model_version
ORDER BY prediction_date DESC;

-- Insert some example data for testing
INSERT INTO monitoring.model_metrics (model_name, model_version, metric_name, metric_value) VALUES
('churn-predictor', '1.0.0', 'accuracy', 0.85),
('churn-predictor', '1.0.0', 'precision', 0.82),
('churn-predictor', '1.0.0', 'recall', 0.78),
('churn-predictor', '1.0.0', 'f1_score', 0.80),
('churn-predictor', '1.0.0', 'auc_roc', 0.88);

COMMIT;