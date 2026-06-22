# ================= BASE IMAGE =================
FROM python:3.11-slim

# ================= SYSTEM SETUP =================
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ================= COPY PROJECT =================
COPY .. /app

# ================= PYTHON SETUP =================
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ================= ENV =================
ENV PYTHONUNBUFFERED=1

# ================= PORT =================
EXPOSE 8000

# ================= START =================
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]