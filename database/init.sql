-- Script de inicialização do banco de dados PostgreSQL
-- Criado automaticamente quando o container PostgreSQL inicia

-- Criar schema para dados do heart disease
CREATE SCHEMA IF NOT EXISTS heart_disease;

-- Tabela para armazenar dados brutos
CREATE TABLE IF NOT EXISTS heart_disease.raw_data (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    sex INTEGER,
    chest_pain_type INTEGER,
    resting_bp INTEGER,
    cholesterol INTEGER,
    fasting_bs INTEGER,
    resting_ecg INTEGER,
    max_hr INTEGER,
    exercise_angina INTEGER,
    oldpeak DECIMAL(3,1),
    st_slope INTEGER,
    target INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela para armazenar predições
CREATE TABLE IF NOT EXISTS heart_disease.predictions (
    id SERIAL PRIMARY KEY,
    patient_data JSONB,
    prediction INTEGER,
    probability DECIMAL(5,4),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela para armazenar métricas dos modelos
CREATE TABLE IF NOT EXISTS heart_disease.model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision_class_0 DECIMAL(5,4),
    precision_class_1 DECIMAL(5,4),
    recall_class_0 DECIMAL(5,4),
    recall_class_1 DECIMAL(5,4),
    f1_class_0 DECIMAL(5,4),
    f1_class_1 DECIMAL(5,4),
    roc_auc DECIMAL(5,4),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para melhorar performance
CREATE INDEX idx_predictions_created_at ON heart_disease.predictions(created_at);
CREATE INDEX idx_predictions_model ON heart_disease.predictions(model_name, model_version);
CREATE INDEX idx_raw_data_created_at ON heart_disease.raw_data(created_at);

-- Comentários nas tabelas
COMMENT ON TABLE heart_disease.raw_data IS 'Dados brutos recebidos via API';
COMMENT ON TABLE heart_disease.predictions IS 'Predições realizadas pelos modelos';
COMMENT ON TABLE heart_disease.model_metrics IS 'Métricas de avaliação dos modelos treinados';

-- Grant permissions (MLFlow precisa criar suas próprias tabelas)
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA heart_disease TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA heart_disease TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA heart_disease TO postgres;

-- Mensagem de sucesso
DO $$
BEGIN
    RAISE NOTICE '✅ Database initialized successfully!';
    RAISE NOTICE '📊 Schema heart_disease created';
    RAISE NOTICE '📋 Tables: raw_data, predictions, model_metrics';
END $$;