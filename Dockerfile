# Use a slim Python image for M4 ARM64 compatibility
FROM python:3.11-slim

# 1. Install system dependencies for Kreuzberg and Postgres
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Install Python dependencies
# Override with --build-arg PIP_INDEX_URL=<mirror> when pypi.org is unreliable.
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 --index-url ${PIP_INDEX_URL} -r requirements.txt

# 4. Copy project files
COPY . .

# 5. Run the application
CMD ["python", "-m", "src.main"]