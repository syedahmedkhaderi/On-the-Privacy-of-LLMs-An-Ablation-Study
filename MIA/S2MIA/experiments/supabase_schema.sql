-- =============================================================================
-- Supabase Migration: S2MIA Sequential Experiment Tables
-- =============================================================================
-- Run this SQL in Supabase SQL Editor (Dashboard > SQL Editor > New Query)
-- Creates 3 tables for storing experiment results, per-sample scores, and config.
-- RLS is disabled on all tables.
-- =============================================================================

-- 1. experiment_config: stores global config for each experiment session
CREATE TABLE IF NOT EXISTS experiment_config (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    config_json jsonb NOT NULL,
    llm_models jsonb,
    retriever_models jsonb,
    datasets jsonb,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE experiment_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiment_config FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "allow_all_experiment_config" ON experiment_config;
CREATE POLICY "allow_all_experiment_config" ON experiment_config FOR ALL USING (true) WITH CHECK (true);

-- 2. experiments: one row per experiment run
CREATE TABLE IF NOT EXISTS experiments (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_group text NOT NULL,
    variable_name text NOT NULL,
    llm_model text NOT NULL,
    retriever_model text NOT NULL,
    dataset_name text NOT NULL,
    corpus_size_pct double precision NOT NULL,
    membership_ratio double precision NOT NULL,
    retrieval_k int NOT NULL,
    test_sample_limit int NOT NULL,
    ref_samples_run int,
    eval_samples_run int,
    threshold_bleu double precision,
    threshold_ppl double precision,
    accuracy double precision,
    "precision" double precision,
    recall double precision,
    f1_score double precision,
    roc_auc double precision,
    pr_auc double precision,
    ref_member_bleu_mean double precision,
    ref_nonmember_bleu_mean double precision,
    ref_member_ppl_mean double precision,
    ref_nonmember_ppl_mean double precision,
    duration_seconds double precision,
    status text NOT NULL DEFAULT 'running',
    error_message text,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE experiments ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiments FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "allow_all_experiments" ON experiments;
CREATE POLICY "allow_all_experiments" ON experiments FOR ALL USING (true) WITH CHECK (true);

-- 3. sample_scores: per-sample BLEU/PPL for ROC curves and distribution plots
CREATE TABLE IF NOT EXISTS sample_scores (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id uuid NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    phase text NOT NULL,
    sample_index int NOT NULL,
    label int NOT NULL,
    s_sem double precision,
    ppl_gen double precision,
    predicted_member boolean
);

ALTER TABLE sample_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE sample_scores FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "allow_all_sample_scores" ON sample_scores;
CREATE POLICY "allow_all_sample_scores" ON sample_scores FOR ALL USING (true) WITH CHECK (true);

-- Index for fast lookups by experiment_id
CREATE INDEX IF NOT EXISTS idx_sample_scores_experiment_id ON sample_scores(experiment_id);

-- Index for filtering experiments by group
CREATE INDEX IF NOT EXISTS idx_experiments_group ON experiments(experiment_group);
