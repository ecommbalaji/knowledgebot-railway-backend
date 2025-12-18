#### Knowledgebase Ingestion
```
POST /api/v1/knowledgebase/upload
Content-Type: multipart/form-data
Body: file (binary), display_name (optional string)

GET /api/v1/knowledgebase/files

DELETE /api/v1/knowledgebase/files/{file_name}
```