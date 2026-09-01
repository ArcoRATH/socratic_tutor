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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project files
COPY . .

# 5. Run the application
CMD ["python", "-m", "src.main"]