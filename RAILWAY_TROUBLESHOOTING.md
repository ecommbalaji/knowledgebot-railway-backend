# Railway Troubleshooting Guide

## Common Errors and Solutions

### Error: "Could not find root directory: api-gateway"

**Problem**: Railway is looking for a directory with a hyphen, but the actual directory uses an underscore.

**Solution**: 

1. Go to Railway Dashboard → Your Project → **api-gateway** service
2. Click on **Settings** tab
3. Find **Root Directory** field
4. Change it from `api-gateway` to `api_gateway` (use underscore, not hyphen)
5. Save and redeploy

**Root Directory Values for All Services**:
- **API Gateway**: `api_gateway` (underscore)
- **Knowledgebase Ingestion**: `services/knowledgebase_ingestion`
- **Website Scraping**: `services/website_scraping`
- **Chatbot Orchestration**: `services/chatbot_orchestration`

**Note**: Service names use hyphens (e.g., `api-gateway`), but root directories use underscores (e.g., `api_gateway`).

---

### Error: "Project Token not found" or "Unauthorized"

**Problem**: Railway CLI authentication issue in GitHub Actions.

**Solution**:

1. Go to Railway Dashboard → Your Project → **Settings** → **Tokens**
2. Create a **PROJECT TOKEN** (not an Account token)
3. Copy the token
4. In GitHub: Go to **Settings** → **Secrets and variables** → **Actions**
5. Add a secret named `RAILWAY_TOKEN` with the project token value
6. Redeploy

---

### Error: Service Build Fails with "No module named X"

**Problem**: Dependencies not installing correctly.

**Solution**:

1. Check `requirements.txt` is in the project root
2. Verify Dockerfile copies `requirements.txt` before installing
3. Check Railway build logs for specific package errors
4. Ensure all dependencies are compatible with Python 3.11

---

### Error: Services Can't Communicate

**Problem**: Internal service URLs not configured correctly.

**Solution**:

1. Verify service names match exactly in Railway (use hyphens: `api-gateway`, `knowledgebase-ingestion`, etc.)
2. Use internal service URLs in environment variables:
   - `http://knowledgebase-ingestion:8001` (not `localhost:8001`)
   - `http://website-scraping:8002`
   - `http://chatbot-orchestration:8003`
3. Ensure services are in the same Railway project
4. Check Railway's internal networking is enabled

---

### Error: Port Already in Use

**Problem**: Multiple services trying to use the same port.

**Solution**:

1. Each service should use its designated port:
   - API Gateway: 8000
   - Knowledgebase Ingestion: 8001
   - Website Scraping: 8002
   - Chatbot Orchestration: 8003
2. Check Dockerfile `EXPOSE` directives match the service ports
3. Verify environment variables like `API_GATEWAY_PORT` match

---

### Error: Build Succeeds but Service Won't Start

**Problem**: Application startup issues.

**Solution**:

1. Check Railway service logs for startup errors
2. Verify all required environment variables are set
3. Test locally first: `docker-compose up`
4. Check Python path issues in Dockerfile
5. Verify main.py is in the correct location

---

### Free Tier Resource Limits

**Problem**: Running out of free tier resources.

**Solution**:

1. Monitor usage in Railway Dashboard → Usage
2. Each service should use defaults:
   - CPU: 0.5 vCPU
   - Memory: 512 MB
3. Total across 4 services: 2.0 vCPU, 2.0 GB RAM
4. If exceeded, consider:
   - Optimizing Docker images
   - Reducing service count
   - Upgrading Railway plan

---

## Quick Checklist

Before deploying, ensure:

- [ ] All services have correct Root Directory (underscores, not hyphens)
- [ ] All environment variables are set (see `RAILWAY_CONFIG.md`)
- [ ] Service names use hyphens (e.g., `api-gateway`)
- [ ] Root directories use underscores (e.g., `api_gateway`)
- [ ] Internal service URLs use service names with hyphens
- [ ] Dockerfiles are in the correct locations
- [ ] `requirements.txt` includes all dependencies
- [ ] Railway project token is set as `RAILWAY_TOKEN` in GitHub Secrets

---

## Getting Help

1. Check Railway build/deploy logs
2. Check service runtime logs in Railway dashboard
3. Test locally with `docker-compose up`
4. Review `RAILWAY_CONFIG.md` for configuration details
5. Check GitHub Actions logs for deployment issues
