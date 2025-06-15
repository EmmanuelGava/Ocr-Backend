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

# Copiar el código de la aplicación y requirements.txt
COPY . .

# Primero, instalar numpy específicamente para asegurar la versión compatible.
# Esto es una solución para forzar la versión si otras dependencias intentan instalar una más nueva.
RUN pip install --no-cache-dir "numpy==1.26.4"

# Luego, instalar el resto de las dependencias desde requirements.txt.
# Esto permitirá que pip use la versión de numpy ya instalada.
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
