FROM python:3.11-slim

# Install Tesseract
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Render sẽ dùng port này)
EXPOSE 10000

# Run Flask app
CMD ["python", "app.py"]
