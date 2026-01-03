# Verification Scripts

## verify_gemini_consistency.py

This script verifies consistency between the PostgreSQL database and Gemini FileSearch.

### Purpose

- Lists all files in the database (from both `file_uploads` and `scraped_websites` tables)
- Verifies each file exists in Gemini FileSearch
- Identifies orphaned files (in database but not in Gemini)
- Identifies missing files (in Gemini but not in database)
- Provides cleanup recommendations

### Usage

```bash
# From the backend root directory
cd /path/to/knowledgebot-railway-backend
python scripts/verify_gemini_consistency.py
```

### Requirements

- Environment variables:
  - `GEMINI_API_KEY`: Your Gemini API key
  - `RAILWAY_POSTGRES_URL`: PostgreSQL connection URL

### Output

The script will:
1. Display a summary of files in database vs Gemini
2. List orphaned files (with their database IDs for manual cleanup)
3. List missing files (files in Gemini but not tracked in database)

### Example Output

```
============================================================
VERIFICATION SUMMARY
============================================================
Total files in database: 15
Total files in Gemini: 13
Orphaned files (in DB but not in Gemini): 2
Missing files (in Gemini but not in DB): 0

============================================================
ORPHANED FILES (in database but not in Gemini):
============================================================
  - ID: 316e9eb5-0ccb-48ea-a2de-eeefb12884f1
    Gemini Name: files/jabbqc3jpdup
    Original: 10.000vragen_Extracted_4pages.pdf
    Table: file_uploads
    Status: NOT_FOUND
```

