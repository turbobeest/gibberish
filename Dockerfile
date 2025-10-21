# Gibberish - Acoustic File Synchronization
# Production Docker Container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY gibberish/ ./gibberish/
COPY tests/ ./tests/
COPY config.yaml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Create directory for synchronized files
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GIBBERISH_CONFIG=/app/config.yaml
ENV NO_COLOR=0

# Expose audio devices (host audio passthrough required)
# Run with: docker run --device /dev/snd gibberish

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD gibberlan validate || exit 1

# Default command shows help
CMD ["gibberlan", "--help"]

# Usage examples:
# Build: docker build -t gibberish .
# Run CLI: docker run --rm -v $(pwd)/data:/data gibberish gibberlan status /data
# Interactive: docker run --rm -it -v $(pwd)/data:/data --device /dev/snd gibberish /bin/bash
# Initialize: docker run --rm -v $(pwd)/data:/data gibberish gibberlan init /data
# Sync (dry-run): docker run --rm -v $(pwd)/data:/data gibberish gibberlan sync /data --dry-run
# Listen: docker run --rm -v $(pwd)/data:/data --device /dev/snd gibberish gibberlan listen
