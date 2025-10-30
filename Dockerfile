# Use a more efficient base image
FROM python:3.9-slim-buster

# Install system dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Install dependencies and clean up pip cache
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip/* \
    && find /usr/local/lib/python3.9/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.9/site-packages -name "__pycache__" -delete

# Copy only necessary application files
COPY manage.py ./
COPY ai_project ./ai_project
COPY chatbot ./chatbot

# Create necessary directories
RUN mkdir -p /app/chroma_data

# Create an empty 'app' file
RUN touch /app/app

# Expose ports for both services
EXPOSE 8080 8000

# Set environment variables
ENV CHROMA_DB_IMPL=duckdb+parquet \
    PERSIST_DIRECTORY=/app/chroma_data \
    DJANGO_SETTINGS_MODULE=ai_project.settings \
    PYTHONUNBUFFERED=1 \
    CHROMA_SERVER_HOST=0.0.0.0 \
    CHROMA_SERVER_PORT=8000

# Create startup script to run both services in parallel
RUN echo '#!/bin/bash\n\
    # Start ChromaDB in the background on port 8000\n\
    chroma run --path /app/chroma_data --host 0.0.0.0 --port 8000 &\n\
    \n\
    # Start Django on port 8080\n\
    python manage.py migrate\n\
    python manage.py runserver 0.0.0.0:8080\n\
    ' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"] 