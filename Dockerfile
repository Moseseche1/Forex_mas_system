FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including MT5 requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install MetaTrader5 dependencies
RUN apt-get update && apt-get install -y \
    libfreetype6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/system/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
