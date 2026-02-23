<p align="center">
  <img src="https://raw.githubusercontent.com/getbindu/create-bindu-agent/refs/heads/main/assets/light.svg" alt="bindu Logo" width="200">
</p>

<h1 align="center">YouTube Video Analysis Agent</h1>
<h3 align="center">AI-Powered Video Content Extraction & Structured Summaries</h3>

<p align="center">
  <strong>Transform YouTube videos into structured, searchable content with accurate timestamps. Extract comprehensive insights, create study guides, and analyze video content with AI-powered precision.</strong><br/>
  Perfect for researchers, students, content creators, and anyone who needs to extract value from video content efficiently.
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/youtube-agent/actions/workflows/build-and-push.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/Paraschamoli/youtube-agent/build-and-push.yml?branch=main" alt="Build status">
  </a>
  <a href="https://img.shields.io/github/license/Paraschamoli/youtube-agent">
    <img src="https://img.shields.io/github/license/Paraschamoli/youtube-agent" alt="License">
  </a>
  <a href="https://img.shields.io/badge/python-3.12-blue">
    <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
  </a>
</p>

---

## 🎯 What is YouTube Video Analysis Agent?

An AI-powered agent that automatically analyzes YouTube videos and creates structured summaries with accurate timestamps. It transforms unstructured video content into organized, searchable data using AI-powered analysis and YouTube transcript extraction.

### Key Features
*   **🎬 Video Transcript Extraction** - Automatic extraction of YouTube video transcripts
*   **⏱️ Accurate Timestamp Creation** - Precise timestamps for major topic transitions
*   **📊 Structured Summaries** - Hierarchical organization with chapters and sections
*   **🎯 Content-Type Detection** - Identifies tutorials, lectures, reviews, documentaries
*   **🔍 Key Insights Extraction** - Extracts main themes, demonstrations, examples
*   **🧠 Conversation Memory** - Mem0 integration for context-aware analysis (optional)
*   **🏗️ Hierarchical Organization** - Chapters → Sections → Key Points structure
*   **📱 Multi-Format Support** - Works with various YouTube video types

### Built-in Tools
*   **YouTubeTools** - Video transcript extraction and metadata analysis
*   **Mem0Tools** - Conversation memory for context-aware analysis (optional)
*   **OpenRouter Integration** - Advanced LLM capabilities for content analysis
*   **Structured Output** - Consistent, validated analysis formats

### Analysis Process
1.  **Video Overview** - Extract metadata, identify video type, determine audience
2.  **Content Extraction** - Fetch and analyze full video transcript
3.  **Timestamp Creation** - Create precise timestamps for major transitions
4.  **Structured Organization** - Group segments into logical sections
5.  **Quality Assurance** - Verify accuracy and comprehensive coverage
6.  **Output Formatting** - Present in clear, navigable format

---

> **🌐 Join the Internet of Agents**
> Register your agent at [bindus.directory](https://bindus.directory) to make it discoverable worldwide and enable agent-to-agent collaboration. **It takes 2 minutes and unlocks the full potential of your agent.**

-> [Postman Collection link](https://raahul-1409c5b4-717533.postman.co/workspace/getbindu's-Workspace~44eb7cfe-a752-4114-8a1a-631395f07bf1/collection/50606358-1c8087bb-fdde-48f6-8773-6a664a209364?action=share&creator=50606358)

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Paraschamoli/youtube-agent.git
cd youtube-agent

# Set up virtual environment with uv
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# OPENROUTER_API_KEY=sk-...      # Required: For OpenRouter LLM
# MEM0_API_KEY=sk-...            # Optional: For conversation memory
# MODEL_NAME=openai/gpt-4o       # Optional: Model ID for OpenRouter
```

### 3. Run Locally

```bash
# Start the YouTube analysis agent
python -m youtube_agent

# Or using uv
uv run python -m youtube_agent
```

### 4. Test with Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at: http://localhost:3773
```

---

## 🔧 Configuration

### Environment Variables
Create a `.env` file:

```env
# Required APIs
OPENROUTER_API_KEY=sk-...           # Required for LLM calls

# Optional features
MEM0_API_KEY=sk-...                 # Optional: Conversation memory
MODEL_NAME=openai/gpt-4o            # Model ID for OpenRouter
```

### Port Configuration
Default port: `3773` (can be changed in `agent_config.json`)

---

## 💡 Usage Examples

### Via HTTP API

```bash
curl -X POST http://localhost:3773/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Analyze this video: https://www.youtube.com/watch?v=..."
      }
    ]
  }'
```

### Sample YouTube Analysis Queries
*   "Analyze this video: https://www.youtube.com/watch?v=..."
*   "Create a study guide from this lecture video"
*   "Extract key points from this tutorial with timestamps"
*   "Summarize this documentary with chapter breakdowns"
*   "Analyze this product review and list all features mentioned"
*   "Extract code examples from this programming tutorial"
*   "Create timestamps for all major topics in this educational video"
*   "Analyze this cooking tutorial and extract ingredient lists"
*   "Summarize this business presentation with key takeaways"
*   "Extract workout routines from this fitness video"

### Expected Response Format

```json
{
  "video_url": "https://www.youtube.com/watch?v=...",
  "title": "Advanced Python Tutorial",
  "duration": "45:30",
  "upload_date": "2024-01-15",
  "video_type": "tutorial",
  "target_audience": "Intermediate Python developers",
  "difficulty_level": "Intermediate",
  "main_themes": ["Asynchronous Programming", "Decorators", "Context Managers"],
  "chapters": [
    {
      "title": "Introduction",
      "start_time": "00:00:00",
      "end_time": "03:45",
      "summary": "Overview of advanced Python concepts covered in the video"
    },
    {
      "title": "Async/Await Deep Dive",
      "start_time": "03:46",
      "end_time": "18:30",
      "summary": "Detailed explanation of async/await patterns with examples",
      "key_points": [
        "How async functions work",
        "Event loop explained",
        "Practical examples"
      ]
    },
    {
      "title": "Advanced Decorators",
      "start_time": "18:31",
      "end_time": "32:15",
      "summary": "Creating parameterized decorators and class decorators",
      "code_examples": [
        "def timer_decorator(func): ...",
        "class CacheDecorator: ..."
      ]
    }
  ],
  "key_demonstrations": [
    {
      "description": "Live coding of async web scraper",
      "timestamp": "12:45",
      "duration": "8 minutes"
    }
  ],
  "resources_mentioned": [
    "Python official documentation",
    "GitHub repository with examples",
    "Recommended books"
  ],
  "practical_takeaways": [
    "Implement async patterns in existing projects",
    "Create reusable decorators",
    "Optimize I/O-bound applications"
  ],
  "content_warnings": null,
  "transcript_quality": "complete",
  "analysis_timestamp": "2024-01-20T10:30:00Z"
}
```

---

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build the image
docker build -t youtube-agent .

# Run container
docker run -d \
  -p 3773:3773 \
  -e OPENROUTER_API_KEY=your_openrouter_key \
  -e MEM0_API_KEY=your_mem0_key \
  -e MODEL_NAME=openai/gpt-4o \
  --name youtube-agent \
  youtube-agent

# Check logs
docker logs -f youtube-agent
```

### Docker Compose (Recommended)

`docker-compose.yml`:

```yaml
version: '3.8'
services:
  youtube-agent:
    build: .
    ports:
      - "3773:3773"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - MEM0_API_KEY=${MEM0_API_KEY}
      - MODEL_NAME=${MODEL_NAME:-openai/gpt-4o}
    restart: unless-stopped
```

### Run with Compose

```bash
# Start with compose
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## 📁 Project Structure

```text
youtube-agent/
├── youtube_agent/
│   ├── skills/
│   │   └── youtube_agent/
│   │       ├── skill.yaml          # Skill configuration
│   │       └── __init__.py
│   ├── __init__.py
│   ├── main.py                     # Agent entry point
│   └── agent_config.json           # Agent configuration
├── agent_config.json               # Bindu agent configuration
├── pyproject.toml                  # Python dependencies
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # Docker Compose setup
├── README.md                       # This documentation
├── .env.example                    # Environment template
└── tests/                          # Test suite
```

---

## 🔌 API Reference

### Health Check

```bash
GET http://localhost:3773/health
```

**Response:**
```json
{"status": "healthy", "agent": "YouTube Video Analysis Agent"}
```

### Chat Endpoint

```bash
POST http://localhost:3773/chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Your YouTube analysis query here"}
  ]
}
```

---

## 🧪 Testing

### Local Testing

```bash
# Install test dependencies
uv sync --group dev

# Run tests
pytest tests/

# Test with coverage
pytest --cov=youtube_agent tests/
```

### Integration Test

```bash
# Start agent
python -m youtube_agent &

# Test API endpoint
curl -X POST http://localhost:3773/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Analyze https://www.youtube.com/watch?v=..."}]}'
```

---

## 🚨 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
| :--- | :--- |
| "OPENROUTER_API_KEY required" | Get your key from openrouter.ai - Required for LLM calls |
| "YouTube video not accessible" | Ensure video has captions enabled or is publicly available |
| "Port 3773 already in use" | Change port in `agent_config.json` or kill the process: `lsof -ti:3773 | xargs kill -9` |
| "Transcript extraction failed" | Some videos may not have available transcripts; try a different video |
| "Memory issues with long videos" | Consider analyzing in sections or using shorter videos |
| "Rate limiting" | Implement delays between requests or upgrade API tier |

---

## 📊 Dependencies

### Core Packages
*   **bindu** - Agent deployment framework
*   **agno** - AI agent framework
*   **agno-tools-youtube** - YouTube transcript extraction
*   **mem0ai** - Memory operations
*   **pydantic** - Structured data validation
*   **python-dotenv** - Environment management

### Development Packages
*   **pytest** - Testing framework
*   **ruff** - Code formatting/linting
*   **pre-commit** - Git hooks

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1.  Fork the repository.
2.  Create a feature branch: `git checkout -b feature/improvement`.
3.  Make your changes following the code style.
4.  Add tests for new functionality.
5.  Commit with descriptive messages.
6.  Push to your fork.
7.  Open a Pull Request.

### Code Style
*   Follow PEP 8 conventions.
*   Use type hints where possible.
*   Add docstrings for public functions.
*   Keep functions focused and small.

---

## 📄 License
MIT License - see LICENSE file for details.

---

## 🙏 Credits & Acknowledgments
*   **Developer:** Paras Chamoli
*   **Framework:** Bindu - Agent deployment platform
*   **Agent Framework:** Agno - AI agent toolkit
*   **Video Analysis:** YouTube transcript extraction
*   **Memory System:** Mem0 - Conversation memory API
*   **LLM Provider:** OpenRouter - Model access

---

## 🔗 Useful Links
*   🌐 [Bindu Directory](https://bindus.directory)
*   📚 [Bindu Docs](https://docs.getbindu.com)
*   🐙 [GitHub](https://github.com/ParasChamoli/youtube-agent)
*   🎬 [YouTube Data API](https://developers.google.com/youtube/v3)
*   💬 [Bindu Community](https://discord.gg/bindu)

<p align="center">
  <strong>Built with ❤️ by Paras Chamoli</strong><br/>
  <em>Transforming video content into structured, actionable knowledge</em>
</p>

<p align="center">
  <a href="https://github.com/ParasChamoli/youtube-agent/stargazers">⭐ Star on GitHub</a> •
  <a href="https://bindus.directory">🌐 Register on Bindu</a> •
  <a href="https://github.com/ParasChamoli/youtube-agent/issues">🐛 Report Issues</a>
</p>

<p align="center">
  <em>Note: This agent analyzes publicly available YouTube content for educational and research purposes. Always respect content creators' rights, YouTube's Terms of Service, and use responsibly for legitimate analysis needs.</em>
</p>
