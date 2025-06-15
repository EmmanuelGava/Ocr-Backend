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
# Asumo que requirements.txt está en la raíz de tu repo de backend o en /app si lo mueves
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
# Esto copiará la carpeta 'scripts' y su contenido a /app/scripts/
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
# Ajustado para reflejar que backend.py está dentro de la carpeta 'scripts'
CMD ["uvicorn", "scripts.backend:app", "--host", "0.0.0.0", "--port", "8000"]
