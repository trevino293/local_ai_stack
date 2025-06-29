# Local AI Stack

A self-hosted AI chat interface with Ollama, MCP filesystem server, and Flask web UI.

## Features

- **Local LLM hosting** via Ollama with GPU support
- **Context file management** using MCP filesystem server
- **Modern web interface** with dark blue theme
- **Chat history** persistence and retrieval
- **Multi-model support** with easy switching
- **File upload** for context augmentation
- **Real-time status monitoring** for all services

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU + drivers (optional, for GPU acceleration)
- 8GB+ RAM recommended

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-ai-stack.git
cd local-ai-stack
```

2. Create required directories:
```bash
mkdir -p shared-data/context-files
```

3. Start the stack:
```bash
docker-compose up -d
```

4. Pull an Ollama model:
```bash
docker exec ollama-server ollama pull llama2
# Or try other models: mistral, codellama, etc.
```

5. Access the interface at [https://localhost5000](http://localhost:5000)

6. Shutdown the stack:
```bash
docker-compose down
```

Windows WSL Lock-Up De-Bug

```bash
wsl --shutdown

or

taskkill /F /im wslservice.exe
```


## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Flask Web UI  │────▶│  MCP Filesystem  │────▶│ Shared Storage  │
│   (Port 5000)   │     │   (Port 3000)    │     │ /context-files  │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Ollama Server  │
│  (Port 11434)   │
└─────────────────┘
```

## Configuration

### Environment Variables

- `OLLAMA_HOST`: Ollama API endpoint (default: http://ollama:11434)
- `MCP_SERVER_URL`: MCP server endpoint (default: http://mcp-filesystem:3000)
- `FLASK_ENV`: Flask environment (default: production)

### GPU Support

GPU support is enabled by default. To disable:
```yaml
# In docker-compose.yml, comment out:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

## Usage

### Chat Interface
- Select a model from the dropdown
- Type messages and press Send or Ctrl+Enter
- View chat history in the right panel

### Context Files
- Upload .txt, .md, .json, or .csv files
- Select files to include as context for queries
- Delete files when no longer needed

### Model Management
```bash
# List models
docker exec ollama-server ollama list

# Pull new model
docker exec ollama-server ollama pull <model-name>

# Remove model
docker exec ollama-server ollama rm <model-name>
```

## Troubleshooting

### Ollama not responding
```bash
docker logs ollama-server
docker restart ollama-server
```

### GPU not detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Port conflicts
Change ports in docker-compose.yml:
```yaml
ports:
  - "8080:5000"  # Change host port
```

## Development

### Local development without Docker
```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: MCP Server
cd mcp-server
npm install
npm start

# Terminal 3: Flask
cd flask-app
pip install -r requirements.txt
python app.py
```

### Adding new features
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Security Notes

- This stack is designed for local/private network use
- No authentication is implemented by default
- For public deployment, add authentication and HTTPS

## License

MIT License