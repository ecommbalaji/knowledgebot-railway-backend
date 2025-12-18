"""Smoke check helper: query /health endpoints for local services with retries.

Usage:
    python scripts/smoke_check.py

This script reads service URLs from environment variables or uses defaults:
- API_GATEWAY_URL (default http://localhost:8000)
- KNOWLEDGEBASE_INGESTION_URL (default http://localhost:8001)
- WEBSITE_SCRAPING_URL (default http://localhost:8002)
- CHATBOT_ORCHESTRATION_URL (default http://localhost:8003)

It will attempt each /health endpoint with retries and print results.
"""
import os
import time
import sys
import json
from urllib.parse import urljoin

import http.client
import urllib.request

SERVICE_ENV = {
    "api_gateway": os.getenv("API_GATEWAY_URL", "http://localhost:8000"),
    "knowledgebase_ingestion": os.getenv("KNOWLEDGEBASE_INGESTION_URL", "http://localhost:8001"),
    "website_scraping": os.getenv("WEBSITE_SCRAPING_URL", "http://localhost:8002"),
    "chatbot_orchestration": os.getenv("CHATBOT_ORCHESTRATION_URL", "http://localhost:8003"),
}

RETRIES = int(os.getenv("SMOKE_RETRIES", "5"))
DELAY = float(os.getenv("SMOKE_DELAY", "1.5"))
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT", "5"))


def check_health(url: str) -> tuple[bool, str]:
    health_url = urljoin(url if url.endswith("/") else url + "/", "health")
    req = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            try:
                data = json.loads(body)
                status = data.get("status") or data.get("service") or "unknown"
                return True, f"HTTP {resp.getcode()} - {status}"
            except Exception:
                return True, f"HTTP {resp.getcode()} - response: {body[:200]}"
    except Exception as e:
        return False, str(e)


def main():
    results = {}
    for name, url in SERVICE_ENV.items():
        ok = False
        reason = ""
        for attempt in range(1, RETRIES + 1):
            up, msg = check_health(url)
            if up:
                ok = True
                reason = msg
                break
            else:
                reason = msg
                print(f"[{name}] attempt {attempt}/{RETRIES} failed: {msg}")
                time.sleep(DELAY)
        results[name] = {"url": url, "ok": ok, "details": reason}

    all_ok = all(v["ok"] for v in results.values())
    print("\nSmoke check results:")
    for name, info in results.items():
        print(f" - {name}: {'OK' if info['ok'] else 'FAIL'} - {info['details']}")

    if not all_ok:
        print("\nOne or more services are unhealthy. Exit code 2.")
        sys.exit(2)
    print("\nAll services responded healthy. Exit code 0.")
    sys.exit(0)


if __name__ == '__main__':
    main()
