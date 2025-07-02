# LOCAL AI STACK [OL-MCP]

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ██╗      ██████╗  ██████╗ █████╗ ██╗         █████╗ ██╗     │
│   ██║     ██╔═══██╗██╔════╝██╔══██╗██║        ██╔══██╗██║     │
│   ██║     ██║   ██║██║     ███████║██║        ███████║██║     │
│   ██║     ██║   ██║██║     ██╔══██║██║        ██╔══██║██║     │
│   ███████╗╚██████╔╝╚██████╗██║  ██║███████╗   ██║  ██║██║     │
│   ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝  ╚═╝╚═╝     │
│                                                                 │
│   SELF-HOSTED // VECTOR-POWERED // PRIVACY-FIRST               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

[![License: MIT](https://img.shields.io/badge/LICENSE-MIT-000000.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/DOCKER-COMPOSE-000000.svg?style=flat-square)](https://docs.docker.com/compose/)
[![Qdrant](https://img.shields.io/badge/QDRANT-VECTOR_DB-000000.svg?style=flat-square)](https://qdrant.tech)
[![Python](https://img.shields.io/badge/PYTHON-3.11+-000000.svg?style=flat-square)](https://python.org)
[![Node.js](https://img.shields.io/badge/NODE.JS-18+-000000.svg?style=flat-square)](https://nodejs.org)

---

## OVERVIEW

```
WHAT:    Self-hosted AI chat with persistent vector search
WHY:     Complete data privacy with enterprise performance  
HOW:     Docker Compose + Qdrant + Ollama + Flask
WHERE:   Your infrastructure, your control
```

---

## ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   [USER] ──────> :5000 FLASK ────┬───> :11434 OLLAMA       │
│                        │         │                          │
│                        │         └───> :3000 MCP ───┐       │
│                        │                            │       │
│                        └───> :6333 QDRANT <────────┘       │
│                                    │                        │
│                                    └─> PERSISTENT VECTORS   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### CORE COMPONENTS

| SERVICE | PORT | PURPOSE |
|---------|------|---------|
| FLASK | 5000 | Web interface + RAG pipeline |
| MCP | 3000 | File management + embeddings |
| QDRANT | 6333 | Vector database + dashboard |
| OLLAMA | 11434 | LLM inference server |

---

## QUICK START

```bash
# CLONE
git clone https://github.com/trevino293/local_ai_stack.git
cd local_ai_stack

# BUILD
docker-compose up -d

# MODEL
docker exec ollama-server ollama pull llama3.2:3b

# ACCESS
http://localhost:5000      # Main UI
http://localhost:6333      # Qdrant Dashboard
```

### SYSTEM REQUIREMENTS

```
RAM:     8GB minimum
DISK:    10GB for models + vectors  
GPU:     Optional NVIDIA for acceleration
DOCKER:  Latest version required
```

---

## FEATURES

### VECTOR SEARCH
- **384-dimensional embeddings**
- **Cosine similarity ranking**
- **System file prioritization**
- **Persistent storage in Qdrant**
- **Sub-100ms search latency**

### REASONING ENGINE
- **6-stage deliberation pipeline**
- **Confidence scoring [1-10]**
- **Citation tracking**
- **Context injection**
- **Transparent processing**

### PROCESSING MODES
```
FAST MODE:     3 chunks, ~2-3s response
DETAILED MODE: 5+ chunks, ~5-7s response
```

---

## CONFIGURATION

### MODEL PARAMETERS

```
TEMPERATURE     [0.0-2.0]    Randomness control
TOP_P           [0.0-1.0]    Nucleus sampling  
TOP_K           [1-100]      Vocabulary limit
REPEAT_PENALTY  [0.5-2.0]    Repetition control
```

### ENVIRONMENT

```yaml
# docker-compose.yml
OLLAMA_HOST: http://ollama:11434
MCP_SERVER_URL: http://mcp-filesystem:3000  
VECTOR_DB_URL: http://qdrant:6333
COLLECTION_NAME: documents
DEFAULT_MODEL: llama3.2:3b
```

---

## API REFERENCE

### CHAT ENDPOINTS
```
POST   /api/chat              # RAG-powered chat
GET    /api/chat/history      # Conversation history
DELETE /api/chat/history      # Clear history
```

### VECTOR OPERATIONS
```
POST   /search                # Semantic search
GET    /vectors/stats         # Vector statistics
POST   /collections/init      # Initialize collection
```

### FILE MANAGEMENT
```
GET    /api/files             # List with vector status
POST   /api/files             # Upload + vectorize
DELETE /api/files/{filename}  # Delete file + vectors
```

### SYSTEM STATUS
```
GET    /api/system/info       # System information
GET    /api/mcp/status        # MCP server status
GET    /api/embedding/health  # Vector health check
```

---

## USAGE

### BASIC CHAT
```javascript
fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: "What is the system architecture?",
        model: "llama3.2:3b",
        fast_mode: true
    })
});
```

### VECTOR SEARCH
```bash
curl -X POST http://localhost:3000/search \
    -H "Content-Type: application/json" \
    -d '{"query": "vector database", "topK": 5}'
```

### FILE OPERATIONS
```bash
# Upload with auto-vectorization
curl -X POST -F "file=@document.pdf" http://localhost:5000/api/files

# Check vector stats
curl http://localhost:3000/vectors/stats
```

---

## PERFORMANCE

```
┌─────────────────────────────────────────────────┐
│ OPERATION          │ FAST MODE │ DETAILED MODE │
├────────────────────┼───────────┼───────────────┤
│ Simple Query       │ 2-3s      │ 5-7s          │
│ Complex Analysis   │ 4-6s      │ 10-15s        │
│ Vector Search      │ <100ms    │ <200ms        │
│ File Upload (1MB)  │ ~2s       │ ~2s           │
└────────────────────┴───────────┴───────────────┘
```

---

## TROUBLESHOOTING

### QDRANT ISSUES
```bash
docker logs vector-database
curl http://localhost:6333/

# Reset collection
curl -X DELETE http://localhost:6333/collections/documents
curl -X POST http://localhost:3000/collections/init
```

### MODEL ISSUES
```bash
docker exec ollama-server ollama list
docker exec ollama-server ollama pull llama3.2:3b
```

### VECTOR SEARCH
```bash
curl http://localhost:3000/vectors/stats
docker-compose restart mcp-filesystem
```

---

## DEVELOPMENT

```bash
# Local setup
cd flask-app && python app.py
cd mcp-server && npm start
docker run -d -p 6333:6333 qdrant/qdrant

# Testing
python e2e_test.py
python RAG_debugger.py
```

---

## LICENSE

```
MIT License
Copyright (c) 2025 Local AI Stack Contributors

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the 
"Software"), to deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software.
```

---

## BUILT WITH

**[QDRANT](https://qdrant.tech)** // **[OLLAMA](https://ollama.ai)** // **[FLASK](https://flask.palletsprojects.com)** // **[DOCKER](https://docker.com)**

---

```
END OF DOCUMENT
```