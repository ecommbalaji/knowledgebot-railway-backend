-- Create indexes for scraped websites
CREATE INDEX IF NOT EXISTS idx_scraped_websites_user_id ON scraped_websites(user_id);
CREATE INDEX IF NOT EXISTS idx_scraped_websites_domain ON scraped_websites(domain);
CREATE INDEX IF NOT EXISTS idx_scraped_websites_gemini_file_name ON scraped_websites(gemini_file_name);
CREATE INDEX IF NOT EXISTS idx_scraped_websites_scraped_at ON scraped_websites(scraped_at DESC);
CREATE INDEX IF NOT EXISTS idx_scraped_websites_gemini_state ON scraped_websites(gemini_state);