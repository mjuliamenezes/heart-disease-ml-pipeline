-- Script para adicionar apenas a tabela de predições (sem apagar nada)
-- Execute: docker compose exec postgres psql -U postgres -d mlflow_db -f /path/to/this/file.sql

-- Criar tabela de predições se não existir
CREATE TABLE IF NOT EXISTS heart_disease.predictions (
    id SERIAL PRIMARY KEY,
    patient_data JSONB,
    prediction INTEGER,
    probability DECIMAL(5,4),
    true_label INTEGER,  -- Para comparação durante validação
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Criar índices
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON heart_disease.predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON heart_disease.predictions(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON heart_disease.predictions(prediction);

-- Criar views úteis
CREATE OR REPLACE VIEW heart_disease.prediction_distribution AS
SELECT 
    prediction,
    COUNT(*) as count,
    ROUND((COUNT(*)::FLOAT / SUM(COUNT(*)) OVER () * 100)::NUMERIC, 2) as percentage
FROM heart_disease.predictions
GROUP BY prediction;

CREATE OR REPLACE VIEW heart_disease.hourly_stats AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as disease_predictions,
    SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as healthy_predictions,
    ROUND(AVG(probability)::NUMERIC, 4) as avg_probability
FROM heart_disease.predictions
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;

-- View para acurácia (quando true_label estiver disponível)
CREATE OR REPLACE VIEW heart_disease.accuracy_over_time AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(
        (SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT * 100)::NUMERIC,
        2
    ) as accuracy_percentage
FROM heart_disease.predictions
WHERE true_label IS NOT NULL
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;

-- Comentários
COMMENT ON TABLE heart_disease.predictions IS 'Predições realizadas em tempo real via streaming';
COMMENT ON COLUMN heart_disease.predictions.true_label IS 'Label verdadeiro (usado para validação)';

-- Mensagem
DO $$
BEGIN
    RAISE NOTICE '✅ Tabela predictions criada/atualizada com sucesso!';
END $$;