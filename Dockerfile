FROM python:3.9-slim

# Instalar dependencias del sistema requeridas para PaddleOCR y otras librerías
# Esto incluye:
# - libgl1: Para operaciones gráficas de OpenCV.
# - libgomp1: Para el soporte OpenMP de PaddlePaddle (soluciona 'libgomp.so.1').
# - libglib2.0-0: Para dependencias de OpenCV (soluciona 'libgthread-2.0.so.0').
# - poppler-utils: Para pdf2image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar archivos de requisitos e instalar dependencias de Python
# Asumo que requirements.txt está directamente en la raíz de tu repositorio de backend.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
# Esto copiará tu backend.py (y cualquier otro archivo en la raíz del repo) a /app/
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación con Uvicorn
# Asumo que tu archivo principal es 'backend.py' y está directamente en /app/
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
