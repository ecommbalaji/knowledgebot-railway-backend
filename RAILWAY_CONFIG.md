# Railway Configuration Reference

Complete list of environment variables for each service in Railway.

## API Gateway Service

**Service Name**: `api-gateway`  
**Root Directory**: `api_gateway` ⚠️ **IMPORTANT: Use underscore, NOT hyphen**  
**Port**: `8000`  
**Public Domain**: Yes (Generate Domain)  
**CPU**: `0.5 vCPU` (Free Tier default)  
**Memory**: `512 MB` (Free Tier default)

### ⚠️ Common Error Fix:
If you see **"Could not find root directory: api-gateway"**:
- The Root Directory must be: `api_gateway` (with **underscore**, not hyphen)
- Service name is `api-gateway` (with hyphen), but root directory uses underscore
- Update in Railway Dashboard: **Settings** → **Root Directory** → Set to `api_gateway`

### Resource Settings:
- **Settings** → **Resources**:
  - CPU: `0.5 vCPU` (default, keep as is)
  - Memory: `512 MB` (default, keep as is)

### Environment Variables:

```env
KNOWLEDGEBASE_INGESTION_URL=http://knowledgebase-ingestion:8001
WEBSITE_SCRAPING_URL=http://website-scraping:8002
CHATBOT_ORCHESTRATION_URL=http://chatbot-orchestration:8003
API_GATEWAY_PORT=8000
API_GATEWAY_HOST=0.0.0.0
```

### Copy-Paste Format (RAW Editor):

```
KNOWLEDGEBASE_INGESTION_URL=http://knowledgebase-ingestion:8001
WEBSITE_SCRAPING_URL=http://website-scraping:8002
CHATBOT_ORCHESTRATION_URL=http://chatbot-orchestration:8003
API_GATEWAY_PORT=8000
API_GATEWAY_HOST=0.0.0.0
```

---

## Knowledgebase Ingestion Service

**Service Name**: `knowledgebase-ingestion`  
**Root Directory**: `services/knowledgebase_ingestion`  
**Port**: `8001`  
**Public Domain**: No (Private/Internal)  
**CPU**: `0.5 vCPU` (Free Tier default)  
**Memory**: `512 MB` (Free Tier default)

### Resource Settings:
- **Settings** → **Resources**:
  - CPU: `0.5 vCPU` (default, keep as is)
  - Memory: `512 MB` (default, keep as is)

### Environment Variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Copy-Paste Format (RAW Editor):

```
GEMINI_API_KEY=your_gemini_api_key_here
```

**Replace** `your_gemini_api_key_here` with your actual Gemini API key.

---

## Website Scraping Service

**Service Name**: `website-scraping`  
**Root Directory**: `services/website_scraping`  
**Port**: `8002`  
**Public Domain**: No (Private/Internal)  
**CPU**: `0.5 vCPU` (Free Tier default)  
**Memory**: `512 MB` (Free Tier default)

### Resource Settings:
- **Settings** → **Resources**:
  - CPU: `0.5 vCPU` (default, keep as is)
  - Memory: `512 MB` (default, keep as is)

### Environment Variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Copy-Paste Format (RAW Editor):

```
GEMINI_API_KEY=your_gemini_api_key_here
```

**Replace** `your_gemini_api_key_here` with your actual Gemini API key.

---

## Chatbot Orchestration Service

**Service Name**: `chatbot-orchestration`  
**Root Directory**: `services/chatbot_orchestration`  
**Port**: `8003`  
**Public Domain**: No (Private/Internal)  
**CPU**: `0.5 vCPU` (Free Tier default)  
**Memory**: `512 MB` (Free Tier default)

### Resource Settings:
- **Settings** → **Resources**:
  - CPU: `0.5 vCPU` (default, keep as is)
  - Memory: `512 MB` (default, keep as is)

### Environment Variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
CHATBOT_MODEL=gpt-4o
CHATBOT_TEMPERATURE=0.7
CHATBOT_MAX_TOKENS=2000
HUMAN_IN_THE_LOOP_ENABLED=false
HUMAN_IN_THE_LOOP_WEBHOOK_URL=
GEMINI_FILESEARCH_PROJECT_ID=
GEMINI_FILESEARCH_LOCATION=us-central1
KNOWLEDGEBASE_INGESTION_URL=http://knowledgebase-ingestion:8001
WEBSITE_SCRAPING_URL=http://website-scraping:8002
RAILWAY_ENVIRONMENT=production
```

### Copy-Paste Format (RAW Editor):

```
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
CHATBOT_MODEL=gpt-4o
CHATBOT_TEMPERATURE=0.7
CHATBOT_MAX_TOKENS=2000
HUMAN_IN_THE_LOOP_ENABLED=false
HUMAN_IN_THE_LOOP_WEBHOOK_URL=
GEMINI_FILESEARCH_PROJECT_ID=
GEMINI_FILESEARCH_LOCATION=us-central1
KNOWLEDGEBASE_INGESTION_URL=http://knowledgebase-ingestion:8001
WEBSITE_SCRAPING_URL=http://website-scraping:8002
RAILWAY_ENVIRONMENT=production
```

**Replace**:
- `your_gemini_api_key_here` with your actual Gemini API key
- `your_openai_api_key_here` with your actual OpenAI API key

**Optional Variables** (can be left empty):
- `HUMAN_IN_THE_LOOP_WEBHOOK_URL` - Leave empty if not using webhooks
- `GEMINI_FILESEARCH_PROJECT_ID` - Leave empty (not required for basic FileSearch)

---

## Free Tier Resource Allocation

**Total Resources Across All Services**:
- Total CPU: `2.0 vCPU` (4 services × 0.5 vCPU each)
- Total Memory: `2.0 GB` (4 services × 512 MB each)

**Per-Service Resources** (All services use same defaults):
- CPU: `0.5 vCPU` per service
- Memory: `512 MB` per service

**Note**: These are Railway's default free tier allocations. If you need to adjust:
- Go to **Settings** → **Resources** for each service
- Lower values may cause performance issues
- Higher values may exceed free tier limits
- Monitor usage in Railway dashboard

## Quick Setup Checklist

### Service 1: API Gateway
- [ ] Service name: `api-gateway`
- [ ] Root Directory: `api_gateway`
- [ ] Generate Public Domain
- [ ] Resources: CPU 0.5 vCPU, Memory 512 MB (defaults)
- [ ] Set 5 variables (see above)

### Service 2: Knowledgebase Ingestion
- [ ] Service name: `knowledgebase-ingestion`
- [ ] Root Directory: `services/knowledgebase_ingestion`
- [ ] No public domain
- [ ] Resources: CPU 0.5 vCPU, Memory 512 MB (defaults)
- [ ] Set 1 variable: `GEMINI_API_KEY`

### Service 3: Website Scraping
- [ ] Service name: `website-scraping`
- [ ] Root Directory: `services/website_scraping`
- [ ] No public domain
- [ ] Resources: CPU 0.5 vCPU, Memory 512 MB (defaults)
- [ ] Set 1 variable: `GEMINI_API_KEY`

### Service 4: Chatbot Orchestration
- [ ] Service name: `chatbot-orchestration`
- [ ] Root Directory: `services/chatbot_orchestration`
- [ ] No public domain
- [ ] Resources: CPU 0.5 vCPU, Memory 512 MB (defaults)
- [ ] Set 13 variables (see above)

---

## Variable Descriptions

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for FileSearch |
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM (chatbot only) |
| `KNOWLEDGEBASE_INGESTION_URL` | Yes | Internal service URL for knowledgebase service |
| `WEBSITE_SCRAPING_URL` | Yes | Internal service URL for scraping service |
| `CHATBOT_ORCHESTRATION_URL` | Yes | Internal service URL for chatbot service |
| `API_GATEWAY_PORT` | Yes | Port for API Gateway (8000) |
| `API_GATEWAY_HOST` | Yes | Host binding (0.0.0.0) |
| `CHATBOT_MODEL` | Optional | LLM model (default: gpt-4o) |
| `CHATBOT_TEMPERATURE` | Optional | LLM temperature (default: 0.7) |
| `CHATBOT_MAX_TOKENS` | Optional | Max tokens for responses (default: 2000) |
| `HUMAN_IN_THE_LOOP_ENABLED` | Optional | Enable HITL (default: false) |
| `HUMAN_IN_THE_LOOP_WEBHOOK_URL` | Optional | Webhook URL for HITL |
| `GEMINI_FILESEARCH_PROJECT_ID` | Optional | GCP project ID (not required) |
| `GEMINI_FILESEARCH_LOCATION` | Optional | FileSearch location (default: us-central1) |
| `RAILWAY_ENVIRONMENT` | Optional | Environment name (default: production) |

---

## Notes

1. **Service Names**: Must match exactly (case-sensitive, with hyphens)
2. **Ports**: Railway auto-detects from Dockerfile `EXPOSE` directives
3. **Service URLs**: Use service names with hyphens, not underscores
4. **Empty Values**: Variables with empty values can be left as `KEY=` or omitted
5. **RAW Editor**: Use the RAW Editor in Railway Variables tab for bulk entry
6. **Resources**: All services use Railway defaults (0.5 vCPU, 512 MB) which are within free tier limits
7. **Resource Location**: Configure resources at **Settings** → **Resources** for each service

## Free Tier Limits Reference

**Railway Free Tier** typically includes:
- Limited compute hours per month
- Default resource allocation per service: 0.5 vCPU, 512 MB RAM
- Multiple services allowed (our 4 services are within limits)
- Public domains included
- Internal networking between services included

**To verify your usage**:
- Go to Railway dashboard → Project → Usage/Stats
- Monitor resource consumption
- Adjust if needed (but defaults should work fine for MVP)
