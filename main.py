from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import pytesseract
import re
import io
import pandas as pd
from PIL import Image
import pdf2image
import os
import tempfile
import logging

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

# Función para extraer texto de una imagen usando OCR
def extract_text_from_image(image):
    try:
        # Configuración para español
        text = pytesseract.image_to_string(image, lang='spa')
        return text
    except Exception as e:
        logger.error(f"Error en OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en OCR: {str(e)}")

# Función para extraer datos específicos del texto
def extract_invoice_data(text):
    # Inicializar diccionario de datos
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
    fecha_pattern = r'(?:FECHA|Date)[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})'
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
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Archivo recibido: {file.filename}")
        
        # Verificar el tipo de archivo
        content_type = file.content_type
        if not (content_type.startswith("image/") or content_type == "application/pdf"):
            raise HTTPException(status_code=400, detail="Tipo de archivo no soportado. Solo se aceptan imágenes (JPEG, PNG) o PDFs.")
        
        # Crear un archivo temporal para guardar el contenido
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="El archivo está vacío")
            
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Procesar según el tipo de archivo
            if content_type == "application/pdf":
                logger.info("Procesando PDF...")
                # Convertir PDF a imágenes
                images = pdf2image.convert_from_path(temp_path)
                if not images:
                    raise HTTPException(status_code=400, detail="No se pudieron extraer imágenes del PDF")
                # Procesar solo la primera página por simplicidad
                text = extract_text_from_image(images[0])
            else:  # Imagen
                logger.info("Procesando imagen...")
                image = Image.open(temp_path)
                text = extract_text_from_image(image)
            
            # Extraer datos de la factura
            logger.info("Extrayendo datos...")
            invoice_data = extract_invoice_data(text)
            
            # Crear DataFrame y Excel
            df = pd.DataFrame([invoice_data])
            
            # Crear un buffer en memoria para el Excel
            excel_buffer = io.BytesIO()
            
            # Guardar DataFrame como Excel
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Datos Factura')
            
            # Preparar el buffer para lectura
            excel_buffer.seek(0)
            
            # Nombre del archivo
            filename = f"datos_factura_{file.filename.split('.')[0]}.xlsx"
            
            # Devolver el archivo Excel como respuesta
            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            }
            
            # Usar Response en lugar de StreamingResponse para tener más control
            return Response(
                content=excel_buffer.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers=headers
            )
        finally:
            # Asegurarse de eliminar el archivo temporal
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except HTTPException as e:
        # Re-lanzar excepciones HTTP para que sean manejadas por el manejador personalizado
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

# Para probar localmente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
