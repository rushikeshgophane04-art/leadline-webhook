# Use slim Python base
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps required by google libs and ffmpeg if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port and default env
ENV PORT=8080

# Use gunicorn for production
CMD exec gunicorn main:app --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 300
