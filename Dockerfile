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
RUN pip install --no-cache-dir "numpy==1.26.4"  # Forzar la versión de numpy
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Paso para pre-descargar los modelos de PaddleOCR durante la construcción
# Se añade KMP_DUPLICATE_LIB_OK para resolver posibles conflictos OpenMP
# Se añaden print statements para mayor verbosidad en los logs de construcción
RUN python -c "import os; os.environ[\'KMP_DUPLICATE_LIB_OK\']=\'TRUE\'; from paddleocr import PaddleOCR; print(\'Intentando inicializar PaddleOCR para pre-descarga de modelos...\'); _=PaddleOCR(use_angle_cls=True, lang=\'es\', use_gpu=False); print(\'PaddleOCR: Modelos pre-descargados exitosamente.\')"

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
