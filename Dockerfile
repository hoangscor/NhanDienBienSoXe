FROM python:3.11-slim

# Cài các thư viện hệ thống cần thiết
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Tạo thư mục app
WORKDIR /app

# Copy toàn bộ code vào container
COPY . /app

# Cài pip và các thư viện Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Khởi chạy app
CMD ["python", "app.py"]
