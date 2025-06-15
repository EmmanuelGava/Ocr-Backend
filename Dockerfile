FROM python:3.9-slim

# Instalar dependencias del sistema, incluyendo libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar archivos de requisitos e instalar dependencias
# Asumo que requirements.txt está en la raíz del repo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
# Esto copiará backend.py (y cualquier otro archivo en la raíz del repo) a /app/
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
# Ajustado para reflejar que backend.py está directamente en la raíz del WORKDIR /app
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
