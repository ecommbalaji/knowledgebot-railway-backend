# Knowledge Bot Railway Backend

A knowledge-based chatbot backend built with Pydantic AI, designed for deployment on Railway's free tier. This MVP provides RAG (Retrieval-Augmented Generation) capabilities over content scraped from websites using Gemini FileSearch for embeddings and OpenAI GPT-4o for orchestration.

## Architecture

The backend consists of four main services:

1. **API Gateway** (`api_gateway/`): Central entry point routing requests to appropriate services
2. **Knowledgebase Ingestion Service** (`services/knowledgebase_ingestion/`): Handles document uploads to Gemini FileSearch
3. **Website Scraping Service** (`services/website_scraping/`): Scrapes websites using Crawl4AI and uploads to Gemini FileSearch
4. **Chatbot Orchestration Service** (`services/chatbot_orchestration/`): Pydantic AI agent with RAG, chat history, and human-in-the-loop support

## Features

- **Pydantic AI Integration**: Full-featured agent with type safety, structured outputs, tool calling, dependency injection, and self-correction
- **RAG with Gemini FileSearch**: Semantic search over uploaded documents and scraped content
- **In-Memory Session Management**: Ephemeral session storage suitable for Railway free tier
- **Human-in-the-Loop**: Configurable endpoints for manual review and intervention
- **Model Agnostic**: Built with Pydantic AI for easy LLM model switching
- **Railway Optimized**: Lightweight Docker images, minimal resource usage

## Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Railway account (for deployment)
- API Keys:
  - Google Gemini API key (for FileSearch)
  - OpenAI API key (for LLM)

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required variables:
- `GEMINI_API_KEY`: Your Google Gemini API key
- `OPENAI_API_KEY`: Your OpenAI API key

Optional variables:
- `CHATBOT_MODEL`: LLM model (default: `gpt-4o`)
- `CHATBOT_TEMPERATURE`: Temperature for LLM (default: `0.7`)
- `HUMAN_IN_THE_LOOP_ENABLED`: Enable human review (default: `false`)
- `HUMAN_IN_THE_LOOP_WEBHOOK_URL`: Webhook URL for reviews

## Local Development

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd knowledgebot-railway-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (see above)

### Running Services Locally

#### API Gateway
```bash
cd api_gateway
python main.py
```
Runs on http://localhost:8000

#### Knowledgebase Ingestion Service
```bash
cd services/knowledgebase_ingestion
python main.py
```
Runs on http://localhost:8001

#### Website Scraping Service
```bash
cd services/website_scraping
python main.py
```
Runs on http://localhost:8002

#### Chatbot Orchestration Service
```bash
cd services/chatbot_orchestration
python main.py
```
Runs on http://localhost:8003

### Using Docker Compose (Recommended for Local Testing)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api-gateway:
    build:
      context: .
      dockerfile: api_gateway/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - KNOWLEDGEBASE_INGESTION_URL=http://knowledgebase-ingestion:8001
      - WEBSITE_SCRAPING_URL=http://website-scraping:8002
      - CHATBOT_ORCHESTRATION_URL=http://chatbot-orchestration:8003
    env_file:
      - .env

  knowledgebase-ingestion:
    build:
      context: .
      dockerfile: services/knowledgebase_ingestion/Dockerfile
    ports:
      - "8001:8001"
    env_file:
      - .env

  website-scraping:
    build:
      context: .
      dockerfile: services/website_scraping/Dockerfile
    ports:
      - "8002:8002"
    env_file:
      - .env

  chatbot-orchestration:
    build:
      context: .
      dockerfile: services/chatbot_orchestration/Dockerfile
    ports:
      - "8003:8003"
    env_file:
      - .env
```

Run with:
```bash
docker-compose up --build
```

## API Endpoints

### API Gateway (Port 8000)

All requests go through the API Gateway:

#### Health Check
```
GET /health
```

#### Knowledgebase Ingestion
```
POST /api/v1/knowledgebase/upload
Content-Type: multipart/form-data
Body: file (binary), display_name (optional string)

GET /api/v1/knowledgebase/files
```

#### Website Scraping
```
POST /api/v1/scrape
Content-Type: application/json
Body: {
  "url": "https://books.toscrape.com",
  "max_depth": 1,
  "max_pages": 10
}
```

#### Chatbot
```
POST /api/v1/chat
Content-Type: application/json
Body: {
  "message": "What books are available?",
  "session_id": "optional-session-id",
  "use_rag": true,
  "max_results": 5
}

GET /api/v1/chat/sessions
DELETE /api/v1/chat/sessions/{session_id}

POST /api/v1/chat/{session_id}/review
Content-Type: application/json
Body: {
  "approved": true,
  "feedback": "optional feedback",
  "corrected_answer": "optional correction"
}
```

## Deployment to Railway

### Prerequisites
1. Railway account at https://railway.app
2. Railway API token (from Railway → Account Settings → Tokens) - for GitHub Actions deployment
3. GitHub repository

### Manual Setup Steps

#### Step 1: Create Railway Project

1. Go to https://railway.app
2. Click **"New Project"**
3. Select **"Empty Project"** or **"Deploy from GitHub repo"**
4. Note your project ID from the URL: `https://railway.app/project/{PROJECT_ID}`

#### Step 2: Create API Gateway Service

1. **Create the Service**:
   - In your Railway project dashboard, click **"+ New"** or **"New Service"**
   - Select **"GitHub Repo"**
   - Choose your repository from the list
   - Railway will create a service (it may have a default name)

2. **Rename the Service**:
   - Click on the service name at the top (or in the service card)
   - Click the edit/pencil icon next to the service name
   - Rename it to exactly: `api-gateway`
   - Press Enter or click Save

3. **Configure Build Settings**:
   - Click on the service to open it
   - Go to **"Settings"** tab (or click the gear icon)
   - Scroll to **"Build & Deploy"** section
   - Find **"Root Directory"** field
   - Set it to: `api_gateway`
   - The **"Dockerfile Path"** should auto-detect as `Dockerfile` (leave as is)

4. **Configure Networking & Port**:
   - In **Settings** tab, scroll to **"Networking"** section
   - **Port**: Railway auto-detects from Dockerfile `EXPOSE 8000`, but if needed:
     - You can add environment variable `PORT=8000` in Variables tab
     - Or set it in Networking section if the option is visible
   - **Public Domain**: Click **"Generate Domain"** button
     - This creates a public URL like `api-gateway-xxxx.up.railway.app`
     - Copy this URL - this is your public API endpoint

5. **Set Environment Variables**:
   - Go to **"Variables"** tab (top menu or sidebar)
   - Click **"+ New Variable"** for each variable
   - Add these variables one by one:
     ```
     KNOWLEDGEBASE_INGESTION_URL=http://knowledgebase-ingestion:8001
     WEBSITE_SCRAPING_URL=http://website-scraping:8002
     CHATBOT_ORCHESTRATION_URL=http://chatbot-orchestration:8003
     API_GATEWAY_PORT=8000
     API_GATEWAY_HOST=0.0.0.0
     ```
   - Or use **"RAW Editor"** (button in Variables tab) to paste all at once:
     ```
     KNOWLEDGEBASE_INGESTION_URL=http://knowledgebase-ingestion:8001
     WEBSITE_SCRAPING_URL=http://website-scraping:8002
     CHATBOT_ORCHESTRATION_URL=http://chatbot-orchestration:8003
     API_GATEWAY_PORT=8000
     API_GATEWAY_HOST=0.0.0.0
     ```

6. **Resource Configuration** (Free Tier - optional):
   - In **Settings** → **Resources**
   - CPU: Leave as default (0.5 vCPU)
   - Memory: Leave as default (512MB)

#### Step 3: Create Knowledgebase Ingestion Service

1. **Create the Service**:
   - In your Railway project, click **"+ New"** or **"New Service"**
   - Select **"GitHub Repo"**
   - Choose the same repository

2. **Rename the Service**:
   - Rename it to exactly: `knowledgebase-ingestion`

3. **Configure Build Settings**:
   - Go to **Settings** → **Build & Deploy**
   - Set **Root Directory** to: `services/knowledgebase_ingestion`
   - **Dockerfile Path** should be: `Dockerfile` (auto-detected)

4. **Configure Networking**:
   - Go to **Settings** → **Networking**
   - **Port**: Railway will auto-detect from Dockerfile (8001)
   - If needed, add environment variable `PORT=8001` in Variables tab
   - **Do NOT** generate a public domain (keep this service private/internal)

5. **Set Environment Variables**:
   - Go to **Variables** tab
   - Add:
     ```
     GEMINI_API_KEY=your_actual_gemini_api_key_here
     ```

6. **Resource Configuration**:
   - Leave defaults (0.5 vCPU, 512MB RAM)

#### Step 4: Create Website Scraping Service

1. **Create the Service**:
   - Click **"+ New"** → **"GitHub Repo"** → Select your repository

2. **Rename the Service**:
   - Rename to exactly: `website-scraping`

3. **Configure Build Settings**:
   - **Settings** → **Build & Deploy**
   - **Root Directory**: `services/website_scraping`
   - **Dockerfile Path**: `Dockerfile` (auto-detected)

4. **Configure Networking**:
   - **Settings** → **Networking**
   - **Port**: Auto-detected from Dockerfile (8002), or add `PORT=8002` variable
   - **Do NOT** generate public domain (keep private)

5. **Set Environment Variables**:
   - **Variables** tab:
     ```
     GEMINI_API_KEY=your_actual_gemini_api_key_here
     ```

#### Step 5: Create Chatbot Orchestration Service

1. **Create the Service**:
   - Click **"+ New"** → **"GitHub Repo"** → Select your repository

2. **Rename the Service**:
   - Rename to exactly: `chatbot-orchestration`

3. **Configure Build Settings**:
   - **Settings** → **Build & Deploy**
   - **Root Directory**: `services/chatbot_orchestration`
   - **Dockerfile Path**: `Dockerfile` (auto-detected)

4. **Configure Networking**:
   - **Settings** → **Networking**
   - **Port**: Auto-detected from Dockerfile (8003), or add `PORT=8003` variable
   - **Do NOT** generate public domain (keep private)

5. **Set Environment Variables**:
   - **Variables** tab - Add all of these:
     ```
     GEMINI_API_KEY=your_actual_gemini_api_key_here
     OPENAI_API_KEY=your_actual_openai_api_key_here
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
   - **Note**: Use **RAW Editor** for easier bulk entry:
     - Click **"RAW Editor"** button in Variables tab
     - Paste all variables in format: `KEY=value` (one per line)
     - Empty values are fine (leave as `KEY=` or `KEY=`)
     - Click **"Update Variables"**

6. **Resource Configuration**:
   - Leave defaults for free tier

### Important Notes About Ports

**Railway handles ports automatically**:
- Railway reads the `EXPOSE` directive from your Dockerfile
- Your Dockerfiles already have `EXPOSE 8000`, `EXPOSE 8001`, etc.
- Railway will automatically use these ports

**If you need to explicitly set ports**:
- Add environment variable `PORT=<port_number>` in the Variables tab
- Example: For API Gateway, add `PORT=8000`
- But this is usually not necessary as Railway auto-detects from Dockerfile

**Service-to-service communication**:
- Services communicate using service names: `http://service-name:port`
- The port numbers in the URLs must match the EXPOSE port in Dockerfile
- Railway DNS resolves service names automatically within the project

#### Step 6: Configure GitHub Actions for Automated Deployment

**Important Note**: 
- GitHub Actions **does NOT** create services or set environment variables
- GitHub Actions **only deploys code** after services are manually created
- You must manually create all 4 services and set all environment variables in Railway dashboard (Steps 2-5 above)
- Once services exist, GitHub Actions will automatically deploy code updates on every push to `main`

1. **Add GitHub Secrets**:
   - Go to your GitHub repository → Settings → Secrets and variables → Actions
   - Click **"New repository secret"**
   - Add:
     - Name: `RAILWAY_API_TOKEN`
     - Value: Your Railway API token (from Railway → Account Settings → Tokens)
   - Optional (for better workflow):
     - Name: `RAILWAY_PROJECT_ID`
     - Value: Your Railway project ID (from project URL)

2. **Verify Workflow**:
   - The `.github/workflows/deploy.yml` workflow will automatically deploy code on push to `main`
   - Or manually trigger from Actions tab → "Deploy to Railway" → "Run workflow"
   - This workflow only deploys existing services - it does NOT create them or set variables

### Railway Configuration Summary

**Service Names** (must match exactly):
- `api-gateway` (public)
- `knowledgebase-ingestion` (private)
- `website-scraping` (private)
- `chatbot-orchestration` (private)

**Root Directories**:
- `api-gateway` → `api_gateway`
- `knowledgebase-ingestion` → `services/knowledgebase_ingestion`
- `website-scraping` → `services/website_scraping`
- `chatbot-orchestration` → `services/chatbot_orchestration`

**Ports**:
- API Gateway: 8000
- Knowledgebase Ingestion: 8001
- Website Scraping: 8002
- Chatbot Orchestration: 8003

**Resource Limits** (Free Tier per service):
- CPU: 0.5 vCPU
- Memory: 512MB
- Storage: 1GB

**Service Networking**:
- API Gateway: Public domain (exposed to internet)
- Other services: Private (accessible via service names)
- Services communicate via: `http://service-name:port`

## Usage Example

### 1. Scrape the Target Website
```bash
curl -X POST http://localhost:8000/api/v1/scrape \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://books.toscrape.com",
    "max_depth": 2,
    "max_pages": 20
  }'
```

### 2. Start a Chat Session
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What books are available on the website?",
    "use_rag": true
  }'
```

### 3. Continue Conversation
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me more about the first book",
    "session_id": "<session-id-from-previous-response>"
  }'
```

## Pydantic AI Features Used

The Chatbot Orchestration Service leverages:

- **AI Agents**: Pydantic AI Agent for conversation management
- **Type Safety**: Pydantic models for structured inputs/outputs
- **Dependency Injection**: Session context injection via dependencies
- **Structured Outputs**: ChatResponse model with validation
- **Self-Correction**: ModelRetry for automatic retry on errors
- **Tool Calling**: search_knowledge_base tool for RAG
- **Chat History**: In-memory message history for context
- **Dynamic Prompts**: System prompts updated with file context
- **Error Handling**: Comprehensive error handling with fallbacks

## Project Structure

```
.
├── api_gateway/
│   ├── main.py
│   └── Dockerfile
├── services/
│   ├── knowledgebase_ingestion/
│   │   ├── main.py
│   │   └── Dockerfile
│   ├── website_scraping/
│   │   ├── main.py
│   │   └── Dockerfile
│   └── chatbot_orchestration/
│       ├── main.py
│       └── Dockerfile
├── shared/
│   ├── config.py
│   └── utils.py
├── .github/
│   └── workflows/
│       └── deploy.yml
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Limitations (MVP)

- In-memory session storage (sessions lost on restart)
- Simplified FileSearch implementation (uses basic file listing)
- No persistent database
- Single-region deployment
- Limited error recovery

## Troubleshooting

### Service Health Checks Failing
- Verify environment variables are set correctly
- Check service URLs match Railway service names
- Ensure API keys are valid

### File Upload Issues
- Verify Gemini API key has FileSearch access
- Check file size limits (Railway free tier: 1GB storage)
- Ensure proper MIME types

### Chatbot Not Responding
- Verify OpenAI API key is valid
- Check model availability (gpt-4o requires API access)
- Review logs for rate limiting issues

### Railway Deployment Issues
- Ensure Dockerfiles use correct context paths
- Verify all environment variables are set in Railway
- Check service resource limits

## References

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Gemini FileSearch API](https://ai.google.dev/gemini-api/docs/file-search)
- [Crawl4AI GitHub](https://github.com/unclecode/crawl4ai)
- [Railway Documentation](https://docs.railway.app/)

## License

See LICENSE file for details.
