# Local AI Stack

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/trevino293/local_ai_stack)](https://github.com/trevino293/local_ai_stack/issues)
[![GitHub stars](https://img.shields.io/github/stars/trevino293/local_ai_stack)](https://github.com/trevino293/local_ai_stack/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/trevino293/local_ai_stack)](https://github.com/trevino293/local_ai_stack/network)
[![Last commit](https://img.shields.io/github/last-commit/trevino293/local_ai_stack)](https://github.com/trevino293/local_ai_stack/commits)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-orange.svg)](https://ollama.ai)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A self-hosted AI chat interface featuring Ollama integration, MCP filesystem server, and an enhanced RAG pipeline with deliberation capabilities.

## ✨ Features

### Core AI Capabilities
- **Enhanced RAG Pipeline** with two-stage deliberation processing
- **Confidence Scoring** and reasoning transparency for all responses
- **Citation Tracking** with automatic source references
- **Multi-Model Support** with dynamic switching between Ollama models

### Interface & User Experience
- **Modern Web Interface** with dark blue theme and responsive design
- **Advanced Parameter Configuration** with presets and saved configurations
- **Context File Management** with system/user file distinction
- **Real-Time Chat History** with enhanced metadata display
- **Collapsible Reasoning Sections** showing AI thought processes

### System Management
- **Containerized Architecture** using Docker Compose
- **MCP Filesystem Server** for context file management
- **Real-Time Status Monitoring** for all services
- **Persistent Configuration Storage** for model parameters

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Flask Web UI  │───▶│  MCP Filesystem  │────▶│ Shared Storage  │
│   (Port 5000)   │     │   (Port 3000)    │     │ /context-files  │
│                 │     │                  │     │                 │
│ • Enhanced RAG  │     │ • File Upload    │     │ • System Files  │
│ • Deliberation  │     │ • CRUD Ops       │     │ • User Files    │
│ • Citations     │     │ • Status API     │     │ • Config Files  │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Ollama Server  │
│  (Port 11434)   │
│                 │
│ • Model Hosting │
│ • GPU Support   │
│ • API Gateway   │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 8GB+ RAM recommended
- NVIDIA GPU + drivers (optional, for acceleration)

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/trevino293/local_ai_stack.git
cd local_ai_stack
```

2. **Create Directories**
```bash
mkdir -p shared-data/context-files
```

3. **Start Services**
```bash
# Build and start all services
docker-compose up -d

# Or rebuild specific services
docker-compose build --no-cache flask-app
docker-compose up -d
```

4. **Install Models**
```bash
# Install a basic model
docker exec ollama-server ollama pull llama3.2

# Or try other models: mistral, codellama, phi3, etc.
docker exec ollama-server ollama list
```

5. **Access Interface**
Open [http://localhost:5000](http://localhost:5000)

6. **Shutdown**
```bash
docker-compose down
```

## 🔧 Configuration

### Environment Variables
```yaml
# In docker-compose.yml
environment:
  - OLLAMA_HOST=http://ollama:11434
  - MCP_SERVER_URL=http://mcp-filesystem:3000
  - FLASK_ENV=production
```

### GPU Support
GPU support is enabled by default. To disable, comment out the deploy section in `docker-compose.yml`:
```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

### Model Parameters
The interface provides three built-in presets:
- **Creative**: High temperature, suitable for brainstorming
- **Balanced**: Default settings for general use
- **Precise**: Low temperature for factual responses

Custom configurations can be saved and reused.

## 📚 Usage

### Enhanced Chat Features
- **Deliberation Mode**: AI shows its reasoning process before responding
- **Confidence Indicators**: 1-10 confidence scores for each response
- **Source Citations**: Automatic tracking of which files influenced responses
- **Context Awareness**: System files are automatically included for consistent behavior

### Context File Management
- Upload `.txt`, `.md`, `.json`, or `.csv` files
- System files (config, admin) are automatically protected and included
- Select specific files for each conversation
- Delete user files when no longer needed

### Model Management
```bash
# List available models
docker exec ollama-server ollama list

# Install new models
docker exec ollama-server ollama pull <model-name>

# Remove models to save space
docker exec ollama-server ollama rm <model-name>
```

## 🔬 Advanced Features

### Two-Stage RAG Pipeline
1. **Deliberation Stage**: Analyzes context relevance and plans response strategy
2. **Response Stage**: Generates final answer with citations and confidence scoring

### Parameter Configuration System
- **Presets**: Quick access to optimized parameter sets
- **Custom Configs**: Save and name your own parameter combinations
- **Model-Specific Settings**: Different parameters per model
- **Real-Time Adjustment**: See effects immediately

### Enhanced Chat History
- **Metadata Tracking**: Model used, parameters, context files, confidence scores
- **Reasoning Preservation**: Full deliberation data stored with each response
- **Filtering Options**: Search by model, confidence level, or date range

## 🛠️ Development

### Local Development (No Docker)
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: MCP Server
cd mcp-server
npm install && npm start

# Terminal 3: Flask Application
cd flask-app
pip install -r requirements.txt
python app.py
```

### Development Setup
- Flask app with hot reload: `FLASK_ENV=development`
- MCP server with nodemon: `npm run dev`
- Frontend changes require browser refresh

## 🔍 Troubleshooting

### Common Issues

**Ollama Not Responding**
```bash
docker logs ollama-server
docker restart ollama-server
```

**GPU Not Detected**
```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**Port Conflicts**
Edit `docker-compose.yml` to change host ports:
```yaml
ports:
  - "8080:5000"  # Change Flask port
  - "8001:3000"  # Change MCP port
```

**WSL Lock-Up (Windows)**
```bash
wsl --shutdown
# or
taskkill /F /im wslservice.exe
```

### Performance Optimization
- Use GPU acceleration when available
- Limit `num_predict` for faster responses
- Reduce context files for simpler queries
- Use precise preset for factual questions

## 🔒 Security Considerations

- **Local Network Only**: Designed for private/local network use
- **No Authentication**: Default setup has no user authentication
- **File Access**: MCP server has full access to `/workspace` directory
- **Production Deployment**: Add HTTPS, authentication, and input validation

For public deployment, implement proper security measures.

## 🧪 API Endpoints

### Chat API
- `POST /api/chat`: Enhanced chat with deliberation
- `GET /api/chat/history`: Retrieve chat history
- `DELETE /api/chat/history`: Clear chat history

### Model Management
- `GET /api/models`: List available models
- `GET /api/model-params`: Get saved parameters
- `POST /api/model-params`: Save parameters

### Configuration Management
- `GET /api/saved-configs`: List saved configurations
- `POST /api/saved-configs`: Create configuration
- `DELETE /api/saved-configs/{id}`: Delete configuration

### File Management
- `GET /api/files`: List context files
- `POST /api/files`: Upload file
- `DELETE /api/files/{filename}`: Delete file

### System Status
- `GET /api/mcp/status`: MCP server status
- `GET /api/system/info`: System information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper testing
4. Submit a pull request with detailed description

### Development Guidelines
- Follow existing code style and structure
- Add comments for complex logic
- Test with multiple models and configurations
- Update documentation for new features

## 📄 License

MIT License

Copyright (c) 2025 Local AI Stack Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM hosting
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Docker](https://www.docker.com/) for containerization
- The open-source AI community for inspiration and tools