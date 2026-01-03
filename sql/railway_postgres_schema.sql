-- Railway PostgreSQL Schema for Knowledge Bot Backend
-- Run this SQL file in your Railway PostgreSQL database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (for tracking who uploaded files)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- File uploads table (stores metadata about uploaded files)
CREATE TABLE IF NOT EXISTS file_uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- File identification
    original_filename VARCHAR(500) NOT NULL,
    display_name VARCHAR(500),
    file_extension VARCHAR(50),

    -- Storage information
    gemini_file_name VARCHAR(500), -- Gemini FileSearch file name/ID
    gemini_file_uri TEXT,

    -- File metadata
    mime_type VARCHAR(255),
    size_bytes BIGINT,
    sha256_hash VARCHAR(64),

    -- Version tracking
    version INTEGER DEFAULT 1, -- Document version (increments on re-upload)

    -- File states
    gemini_upload_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, active, failed
    gemini_state VARCHAR(50), -- ACTIVE, PROCESSING, FAILED, etc.

    -- Timestamps
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    gemini_processed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Website scraping table (stores metadata about scraped websites)
CREATE TABLE IF NOT EXISTS scraped_websites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Website information
    original_url TEXT NOT NULL,
    domain VARCHAR(500),
    title VARCHAR(1000),

    -- Content information
    content_length INTEGER,
    pages_scraped INTEGER DEFAULT 1,

    -- Gemini FileSearch information
    gemini_file_name VARCHAR(500), -- Gemini FileSearch file name/ID
    gemini_file_uri TEXT,

    -- File metadata
    mime_type VARCHAR(255) DEFAULT 'text/markdown',
    size_bytes BIGINT,

    -- Version tracking
    version INTEGER DEFAULT 1, -- Scrape version (increments on re-scrape)

    -- Processing state
    gemini_state VARCHAR(50), -- ACTIVE, PROCESSING, FAILED, etc.

    -- Timestamps
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    gemini_processed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Additional metadata
    scraping_config JSONB DEFAULT '{}'::jsonb, -- max_depth, max_pages, etc.
    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat sessions table (for tracking chat interactions)
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Session metadata
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,

    -- Session state
    is_active BOOLEAN DEFAULT TRUE,
    message_count INTEGER DEFAULT 0,

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages table (stores individual messages in conversations)
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,

    -- Message content
    role VARCHAR(50) NOT NULL, -- user, assistant, system
    content TEXT NOT NULL,

    -- RAG and data source information
    used_rag BOOLEAN DEFAULT FALSE,
    used_postgres BOOLEAN DEFAULT FALSE,
    used_neon_db BOOLEAN DEFAULT FALSE,
    used_internet_search BOOLEAN DEFAULT FALSE,

    -- Response metadata
    confidence_score DECIMAL(3, 2), -- 0.00 to 1.00
    sources JSONB DEFAULT '[]'::jsonb, -- Array of source information
    usage_info JSONB DEFAULT '{}'::jsonb, -- Token usage, etc.

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Metrics table (for tracking system metrics and analytics)
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Metric identification
    metric_type VARCHAR(100) NOT NULL, -- file_upload, chat_request, api_call, etc.
    metric_name VARCHAR(255) NOT NULL,

    -- Metric values
    value DECIMAL(20, 4),
    unit VARCHAR(50), -- bytes, seconds, count, etc.

    -- Context
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    file_upload_id UUID REFERENCES file_uploads(id) ON DELETE SET NULL,

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamp
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API usage tracking table (for monitoring API calls and costs)
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- API identification
    api_provider VARCHAR(100) NOT NULL, -- gemini, openai, cloudflare, etc.
    api_endpoint VARCHAR(255),
    http_method VARCHAR(10),

    -- Request/Response info
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    status_code INTEGER,

    -- Cost tracking (if available)
    cost_usd DECIMAL(10, 6),
    tokens_input INTEGER,
    tokens_output INTEGER,

    -- Context
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,

    -- Timing
    duration_ms INTEGER,

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_file_uploads_user_id ON file_uploads(user_id);
CREATE INDEX IF NOT EXISTS idx_file_uploads_gemini_file_name ON file_uploads(gemini_file_name);
CREATE INDEX IF NOT EXISTS idx_file_uploads_uploaded_at ON file_uploads(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_file_uploads_gemini_state ON file_uploads(gemini_state);
CREATE INDEX IF NOT EXISTS idx_file_uploads_r2_key ON file_uploads(cloudflare_r2_key);

-- Create indexes for scraped websites
CREATE INDEX IF NOT EXISTS idx_scraped_websites_user_id ON scraped_websites(user_id);
CREATE INDEX IF NOT EXISTS idx_scraped_websites_domain ON scraped_websites(domain);
CREATE INDEX IF NOT EXISTS idx_scraped_websites_gemini_file_name ON scraped_websites(gemini_file_name);
CREATE INDEX IF NOT EXISTS idx_scraped_websites_scraped_at ON scraped_websites(scraped_at DESC);
CREATE INDEX IF NOT EXISTS idx_scraped_websites_gemini_state ON scraped_websites(gemini_state);

-- Create indexes for chat tables
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);

-- Create indexes for metrics
CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded_at ON metrics(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_user_id ON metrics(user_id);

-- Create indexes for API usage
CREATE INDEX IF NOT EXISTS idx_api_usage_provider ON api_usage(api_provider);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON api_usage(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
CREATE TRIGGER update_file_uploads_updated_at BEFORE UPDATE ON file_uploads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_scraped_websites_updated_at BEFORE UPDATE ON scraped_websites
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_sessions_updated_at BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert a default system user (optional)
INSERT INTO users (id, email, name, is_active)
VALUES ('00000000-0000-0000-0000-000000000001', 'system@knowledgebot.local', 'System User', TRUE)
ON CONFLICT (id) DO NOTHING;

-- Migration: Add version column if it doesn't exist (for existing databases)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'file_uploads' AND column_name = 'version') THEN
        ALTER TABLE file_uploads ADD COLUMN version INTEGER DEFAULT 1;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'scraped_websites' AND column_name = 'version') THEN
        ALTER TABLE scraped_websites ADD COLUMN version INTEGER DEFAULT 1;
    END IF;
END $$;

-- Migration: Remove R2 storage columns (R2 storage removed - files go directly to Gemini)
DO $$
BEGIN
    -- Remove R2 columns from file_uploads table
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'file_uploads' AND column_name = 'cloudflare_r2_url') THEN
        ALTER TABLE file_uploads DROP COLUMN cloudflare_r2_url;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'file_uploads' AND column_name = 'cloudflare_r2_key') THEN
        ALTER TABLE file_uploads DROP COLUMN cloudflare_r2_key;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'file_uploads' AND column_name = 'r2_upload_status') THEN
        ALTER TABLE file_uploads DROP COLUMN r2_upload_status;
    END IF;
END $$;