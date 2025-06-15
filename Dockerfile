FROM python:3.9-slim

# Instalar dependencias del sistema requeridas para Tesseract OCR, Pillow y pdf2image
# - libgl1: Para operaciones gráficas (OpenCV, Pillow).
# - poppler-utils: Para pdf2image (conversión de PDF a imagen).
# - tesseract-ocr: El motor de Tesseract OCR.
# - tesseract-ocr-spa: Los datos de idioma español para Tesseract.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-spa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar archivos de requisitos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
