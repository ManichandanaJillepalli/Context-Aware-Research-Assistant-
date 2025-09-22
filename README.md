# Context-Aware-Research-Assistant-

🚀 Getting Started
Quick Setup
bash
# Clone and setup
git clone <repository>
cd context-aware-research-assistant

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Add your API keys

# Start services
docker-compose up -d

# Run the application
poetry run uvicorn src.api.routes:app --reload
API Usage
bash
# Create session
curl -X POST "http://localhost:8000/auth/login"

# Search papers
curl -X POST "http://localhost:8000/search" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "neuromorphic computing", "max_results": 10}'

# Complete research pipeline
curl -X POST "http://localhost:8000/research" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "machine learning applications"}'
