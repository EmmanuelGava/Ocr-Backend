FROM python:3.9-slim

# Instalar dependencias del sistema requeridas para PaddleOCR y otras librerías
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar archivos de requisitos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Paso para pre-descargar los modelos de PaddleOCR durante la construcción
# Esto asegura que los modelos estén disponibles cuando la aplicación se inicie
RUN python -c "from paddleocr import PaddleOCR; _=PaddleOCR(use_angle_cls=True, lang='es', use_gpu=False, show_log=False)"

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
