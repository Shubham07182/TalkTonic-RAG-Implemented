# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Install system dependencies (for tesseract, pdfplumber, Pillow, etc.)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "newLLMtotest.py", "--server.port=8501", "--server.address=0.0.0.0"]
