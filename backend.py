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
    logger.info(f"Texto recibido para extracción de datos (primeras 500 chars):\n{text[:500]}")
    # Inicializar diccionario de datos
    data = {
        "cuit_cuil": None,
        "fecha": None,
        "razon_social": None,
        "numero_factura": None,
        "importe_total": None,
        "iva": None
    }
    
    # Patrones de expresiones regulares para extraer datos (mejorados)
    # CUIT/CUIL: 11 dígitos con o sin guiones, o DNI/Pasaporte, o ID Fiscal
    cuit_pattern = r'(?:CUIT|CUIL|C\.U\.I\.T\.|C\.U\.I\.L\.|ID\s*FISCAL|DNI|PASAPORTE)[:\s]*(\b\d{1,2}[-\s]?\d{7,8}[-\s]?\d{1}\b|\b\d{11}\b)'
    
    # Fecha: Ampliar formatos de fecha (DD/MM/AAAA, DD-MM-AAAA, DD.MM.AAAA, AAAA-MM-DD)
    fecha_pattern = r'(?:FECHA|DATE|Fecha|Date|EMISION)[:\s]*(\d{1,2}[/-\.]\d{1,2}[/-\.]\d{2,4}|\d{4}[/-\.]\d{1,2}[/-\.]\d{1,2})'
    
    # Razón Social: Buscar cerca de CUIT/CUIL o con palabras clave comunes
    # Se buscará en un rango de líneas alrededor del CUIT/CUIL o de la palabra "RAZON SOCIAL"
    
    # Número de factura: Ampliar formatos, incluyendo prefijos y sufijos comunes
    factura_pattern = r'(?:FACTURA|FAC|FACTURA\s*N[°º]?|COMPROBANTE|REMITO)[:\s#]*([A-Z]?\s*\d{2,5}[-\s]?\d{6,8}|\d{4}[-\s]?\d{8})'
    
    # Importe total: Más robusto, considerando diferentes escrituras de moneda y separadores
    total_pattern = r'(?:TOTAL|IMPORTE\s*TOTAL|NETO\s*A\s*PAGAR|GRAN\s*TOTAL)[:\s$]*([,\.]+)'
    
    # IVA: Más robusto, considerando diferentes escrituras y separadores
    iva_pattern = r'(?:IVA|I\.V\.A\.|IMPUESTO\s*VALOR\s*AGREGADO|IMPUESTOS)[:\s$]*([,\.]+)'
    
    # Buscar CUIT/CUIL
    cuit_match = re.search(cuit_pattern, text, re.IGNORECASE)
    if cuit_match:
        data["cuit_cuil"] = cuit_match.group(1).strip()
        logger.info(f"CUIT/CUIL encontrado: {data["cuit_cuil"]}")
    
    # Buscar razón social (mejorado)
    # Se buscará en un rango de líneas alrededor del CUIT/CUIL si se encontró, o cerca de "RAZÓN SOCIAL"
    if cuit_match:
        # Extraer texto alrededor del CUIT para buscar la razón social
        start_index = text.find(cuit_match.group(0))
        search_area = text[max(0, start_index - 200):min(len(text), start_index + 200)] # Buscar en 200 chars antes y después
        
        # Patrones para razón social
        razon_social_keywords = ["RAZON SOCIAL", "RAZÓN SOCIAL", "DENOMINACION", "NOMBRE"]
        
        for keyword in razon_social_keywords:
            keyword_upper = keyword.upper()
            lines = search_area.split('\n')
            for i, line in enumerate(lines):
                if keyword_upper in line.upper():
                    # Intentar capturar la línea siguiente como razón social
                    if i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r'^[0-9]+\s*(\.?[0-9]+)?([,\.]\d+)?\s*(%|\$)?$', lines[i+1].strip()): # Evitar números solos
                        data["razon_social"] = lines[i + 1].strip()
                        logger.info(f"Razón Social encontrada (cerca de {keyword}): {data["razon_social"]}")
                        break
                if data["razon_social"]:
                    break
            if data["razon_social"]:
                break
    else:
        # Si no se encontró cerca del CUIT, intentar búsqueda general
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if "RAZÓN SOCIAL" in line.upper() or "RAZON SOCIAL" in line.upper() or "DENOMINACION" in line.upper():
                if i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r'^[0-9]+\s*(\.?[0-9]+)?([,\.]\d+)?\s*(%|\$)?$', lines[i+1].strip()):
                    data["razon_social"] = lines[i + 1].strip()
                    logger.info(f"Razón Social encontrada (búsqueda general): {data["razon_social"]}")
                    break
    
    # Buscar fecha
    fecha_match = re.search(fecha_pattern, text, re.IGNORECASE)
    if fecha_match:
        data["fecha"] = fecha_match.group(1).strip()
        logger.info(f"Fecha encontrada: {data["fecha"]}")
    
    # Buscar número de factura
    factura_match = re.search(factura_pattern, text, re.IGNORECASE)
    if factura_match:
        data["numero_factura"] = factura_match.group(1).strip()
        logger.info(f"Número de factura encontrado: {data["numero_factura"]}")
    
    # Buscar importe total
    total_match = re.search(total_pattern, text, re.IGNORECASE)
    if total_match:
        # Tomar la última coincidencia como el total más probable
        all_total_matches = re.findall(total_pattern, text, re.IGNORECASE)
        if all_total_matches:
            data["importe_total"] = all_total_matches[-1].strip().replace('.', '').replace(',', '.')
            logger.info(f"Importe total encontrado: {data["importe_total"]}")
    
    # Buscar IVA
    iva_match = re.search(iva_pattern, text, re.IGNORECASE)
    if iva_match:
        # Tomar la última coincidencia como el IVA más probable
        all_iva_matches = re.findall(iva_pattern, text, re.IGNORECASE)
        if all_iva_matches:
            data["iva"] = all_iva_matches[-1].strip().replace('.', '').replace(',', '.')
            logger.info(f"IVA encontrado: {data["iva"]}")
    
    logger.info(f"Datos de factura extraídos: {data}")
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
