FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gzip curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY v71_*.py ./
COPY Snakefile config.yaml ./
COPY README.md .

# Port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lancement
CMD ["streamlit", "run", "v71_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
