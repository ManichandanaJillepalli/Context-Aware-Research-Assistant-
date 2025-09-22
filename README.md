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
