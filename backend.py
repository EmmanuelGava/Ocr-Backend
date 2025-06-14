from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import re
import io
import pandas as pd
from PIL import Image
import pdf2image
import os
import tempfile
import logging
import requests
from mistralai.client import MistralClient
from dotenv import load_dotenv
import base64
import json

# Cargar variables de entorno (solo para desarrollo local, en Railway se cargan automáticamente)
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Invoice OCR API")

# Configurar CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes para pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manejadores de excepciones personalizados
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"},
    )

# NUEVO: Obtener API Keys de variables de entorno
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not OCR_SPACE_API_KEY:
    logger.error("OCR_SPACE_API_KEY no está configurada.")
if not MISTRAL_API_KEY:
    logger.error("MISTRAL_API_KEY no está configurada.")

mistral_client = MistralClient(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

# NUEVO: Función para extraer texto de una imagen usando OCR.space
async def perform_ocr_space(image_buffer: bytes, file_type: str) -> str:
    if not OCR_SPACE_API_KEY:
        raise HTTPException(status_code=500, detail="OCR_SPACE_API_KEY no está configurada.")

    # Convertir buffer a base64
    base64_image = base64.b64encode(image_buffer).decode('utf-8')

    payload = {
        "base64Image": f"data:{file_type};base64,{base64_image}",
        "apikey": OCR_SPACE_API_KEY,
        "language": "spa",
        "isOverlayRequired": "false",
        "isTable": "true",  # Para mejorar la extracción de tablas
        "scale": "true",    # Para mejorar la calidad de la imagen antes del OCR
        "detectOrientation": "true" # Para corregir la orientación de la imagen
    }

    try:
        response = requests.post("https://api.ocr.space/parse/image", data=payload)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP 4xx/5xx
        
        ocr_json = response.json()
        logger.info(f"Respuesta de OCR.space: {ocr_json}")

        if ocr_json.get("OCRExitCode") != 1:
            error_message = ocr_json.get("ErrorMessage", ["Error desconocido"])[0]
            raise HTTPException(status_code=500, detail=f"OCR.space falló con código: {ocr_json.get('OCRExitCode')}. Mensaje: {error_message}")
        
        return ocr_json.get("ParsedResults", [{}])[0].get("ParsedText", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al llamar a OCR.space: {e}")
        raise HTTPException(status_code=500, detail=f"Fallo en el servicio OCR (OCR.space): {str(e)}")

# Función para extraer datos específicos del texto (Regex fallback) - RENOMBRADA
def extract_invoice_data_regex(text: str) -> dict:
    data = {
        "cuit_cuil": None,
        "fecha": None,
        "razon_social": None,
        "numero_factura": None,
        "importe_total": None,
        "iva": None
    }
    
    # Patrones de expresiones regulares para extraer datos
    # CUIT/CUIL: 11 dígitos con guiones opcionales
    cuit_pattern = r'(?:CUIT|CUIL|C\.U\.I\.T\.|C\.U\.I\.L\.)[:\s]*([0-9]{2}[-\s]?[0-9]{8}[-\s]?[0-9])'
    # Fecha: formatos comunes de fecha
    fecha_pattern = r'(?:FECHA|Date)[:\s]*(\d{1,2}[/\\-\.]\d{1,2}[/\\-\.]\d{2,4})'
    # Número de factura
    factura_pattern = r'(?:FACTURA|FAC|FACTURA\s+N[°º])[:\s]*([A-Z]?[-\s]?[0-9]{4,8}[-\s]?[0-9]{8})'
    # Importe total
    total_pattern = r'(?:TOTAL|IMPORTE\s+TOTAL)[:\s]*\$?\s*([\d\.,]+)'
    # IVA
    iva_pattern = r'(?:IVA|I\.V\.A\.|IMPUESTO\s+VALOR\s+AGREGADO)[:\s]*\$?\s*([\d\.,]+)'
    
    # Buscar CUIT/CUIL
    cuit_match = re.search(cuit_pattern, text, re.IGNORECASE)
    if cuit_match:
        data["cuit_cuil"] = cuit_match.group(1).strip()
    
    # Buscar fecha
    fecha_match = re.search(fecha_pattern, text, re.IGNORECASE)
    if fecha_match:
        data["fecha"] = fecha_match.group(1).strip()
    
    # Buscar razón social (simplificado - asumimos que está cerca de "CUIT" o "RAZÓN SOCIAL")
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "RAZÓN SOCIAL" in line.upper() or "RAZON SOCIAL" in line.upper():
            if i + 1 < len(lines) and lines[i + 1].strip():
                data["razon_social"] = lines[i + 1].strip()
                break
    
    # Buscar número de factura
    factura_match = re.search(factura_pattern, text, re.IGNORECASE)
    if factura_match:
        data["numero_factura"] = factura_match.group(1).strip()
    
    # Buscar importe total
    total_match = re.search(total_pattern, text, re.IGNORECASE)
    if total_match:
        data["importe_total"] = total_match.group(1).strip().replace('.', '').replace(',', '.')
    
    # Buscar IVA
    iva_match = re.search(iva_pattern, text, re.IGNORECASE)
    if iva_match:
        data["iva"] = iva_match.group(1).strip().replace('.', '').replace(',', '.')
    
    return data

# Añadir endpoint de health check
@app.get("/health")
async def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return {"status": "ok", "message": "API funcionando correctamente"}

# Endpoint raíz para redirigir a la documentación
@app.get("/")
async def root():
    """Endpoint raíz que redirige a la documentación"""
    return {"message": "API de OCR para facturas. Visita /docs para la documentación."}

@app.post("/upload")
async def upload_file(files: list[UploadFile] = File(...)):
    all_extracted_data = []
    
    for file in files:
        try:
            logger.info(f"Archivo recibido: {file.filename}")
            
            # Verificar el tipo de archivo
            content_type = file.content_type
            if not (content_type.startswith("image/") or content_type == "application/pdf"):
                logger.warning(f"Tipo de archivo no soportado para {file.filename}: {content_type}")
                continue
            
            # Leer el contenido del archivo directamente en un buffer
            content_buffer = await file.read()
            if not content_buffer:
                logger.warning(f"El archivo {file.filename} está vacío")
                continue
            
            # Convertir PDF a imagen si es necesario
            processed_image_buffer = None
            if content_type == "application/pdf":
                logger.info("Procesando PDF...")
                # Convertir PDF a imágenes (PIL Image objects)
                images = pdf2image.convert_from_bytes(content_buffer)
                if not images:
                    raise HTTPException(status_code=400, detail=f"No se pudieron extraer imágenes del PDF {file.filename}")
                # Tomar la primera imagen y convertirla a bytes en formato PNG
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='PNG')
                processed_image_buffer = img_byte_arr.getvalue()
                file_type_for_ocr = "image/png"
            else:  # Imagen
                processed_image_buffer = content_buffer
                file_type_for_ocr = content_type

            # --- OCR Step (using OCR.space) ---
            logger.info(f"Realizando OCR para {file.filename} con OCR.space...")
            ocr_text = await perform_ocr_space(processed_image_buffer, file_type_for_ocr)
            logger.info(f"OCR completado para {file.filename}. Texto extraído (primeras 500 chars): {ocr_text[:500]}")

            if not ocr_text.strip():
                logger.warning(f"No se pudo extraer texto de {file.filename}.")
                continue

            extracted_data = {}
            if mistral_client:
                logger.info(f"Enviando texto de {file.filename} a Mistral AI...")
                try:
                    chat_response = mistral_client.chat(
                        model='mistral-large-latest',
                        messages=[
                            {"role": "system", "content": 'You are an AI assistant that extracts structured data from invoice text. Respond only with a JSON object. For each field, if not found, use null. Extract numeric values, converting comma decimals to dot decimals. Be precise with the following fields:\n- CUIT/CUIL: Look for "CUIT", "CUIL", "NIF", or similar identifiers followed by a number. The format is typically XX-XXXXXXXX-X.\n- Company Name (razon_social): This is the legal name of the entity issuing the invoice, often found near the address, CUIT/CUIL, or at the top of the document. Do not confuse it with recipient names.\n- Date: Extract the date of issue.\n- Invoice Number: Look for "FACTURA", "Nº", "FAC", or similar, followed by the invoice number.\n- Total Amount: The final amount of the invoice, usually labeled "TOTAL" or "IMPORTE TOTAL".\n- IVA: The Value Added Tax amount, usually labeled "IVA" or "I.V.A.".\n- Items (productos): Extract a list of product line items. Each item should be an object with the following keys:\n  - codigo: The product code.\n  - producto: The product description.\n  - cantidad: The quantity, as a number.\n  - precio_unitario: The unit price, as a number.\n  - subtotal: The subtotal for that line item, as a number.\n  If no items are found, use an empty array.'},
                            {"role": "user", "content": f"Extract data from this invoice text: \n\n{ocr_text}"}
                        ],
                        response_format={"type": "json_object"}
                    )
                    mistral_output = chat_response.choices[0].message.content
                    logger.info(f"Respuesta de Mistral AI para {file.filename}: {mistral_output}")
                    extracted_data = json.loads(mistral_output)
                except Exception as e:
                    logger.error(f"Error al llamar a Mistral AI o parsear JSON para {file.filename}: {e}")
                    extracted_data = extract_invoice_data_regex(ocr_text)
            else:
                logger.warning("MISTRAL_API_KEY no está configurada. Realizando extracción de datos con expresiones regulares localmente.")
                extracted_data = extract_invoice_data_regex(ocr_text)

            logger.info(f"Datos de factura extraídos para {file.filename}: {extracted_data}")
            all_extracted_data.append(extracted_data)

        except HTTPException as e:
            logger.error(f"Error HTTP procesando {file.filename}: {e.detail}")
        except Exception as e:
            logger.error(f"Error inesperado procesando {file.filename}: {e}")

    if not all_extracted_data:
        raise HTTPException(status_code=400, detail="No se pudieron extraer datos de ningún archivo válido.")

    return JSONResponse(content={"success": True, "data": all_extracted_data}, status_code=200)

# Para probar localmente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
