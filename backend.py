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
    fecha_pattern = r'(?:FECHA|DATE|Fecha|Date|EMISION)(?:\s*de\s*Emisi[oó]n)?:[:\s]*(\d{1,2}[/\\.\-]\d{1,2}[/\\.\-]\d{2,4}|\d{4}[/\\.\-]\d{1,2}[/\\.\-]\d{1,2})'
    
    # Razón Social: Buscar cerca de CUIT/CUIL o con palabras clave comunes
    # Se buscará en un rango de líneas alrededor del CUIT/CUIL o de la palabra "RAZON SOCIAL"
    
    # Número de factura: Ampliar formatos, incluyendo prefijos y sufijos comunes
    factura_pattern = r'(?:FACTURA|FAC|FACTURA\s*N[°º]?|COMPROBANTE|REMITO|Comp\.Nro)[:\s#]*([A-Z]?\s*[\d\s\.\-\/—]+)[\.\s]*'
    
    # Importe total: Más robusto, considerando diferentes escrituras de moneda y separadores
    total_pattern = r'(?:TOTAL|IMPORTE\s*TOTAL|NETO\s*A\s*PAGAR|GRAN\s*TOTAL|Subtotal|Total\s*a\s*Pagar|\$|€|£)[:\s$]*([\d\s\.,]+)'
    
    # IVA: Más robusto, considerando diferentes escrituras y separadores
    iva_pattern = r'(?:IVA|I\.V\.A\.|IMPUESTO\s*VALOR\s*AGREGADO|IMPUESTOS)[:\s$]*([,\.]+)'
    
    # Buscar CUIT/CUIL
    cuit_match = re.search(cuit_pattern, text, re.IGNORECASE)
    if cuit_match:
        data["cuit_cuil"] = cuit_match.group(1).strip()
        logger.info(f"CUIT/CUIL encontrado: {data['cuit_cuil']}")
    
    # Buscar razón social (mejorado)
    # Se buscará en un rango de líneas alrededor del CUIT/CUIL si se encontró, o cerca de "RAZÓN SOCIAL"
    # Priorizar la búsqueda en la misma línea o en las adyacentes
    razon_social_found = False
    
    # Patrones para razón social, incluyendo errores comunes de OCR y términos legales
    # Más específicos para capturar nombres de empresas y evitar fechas/números
    razon_social_patterns = [
        r'(?:RAZ[OÓ]N\s*SOCIAL|DENOMINACI[OÓ]N|NOMBRE)[:\s]*([A-Z0-9][A-Za-z0-9\s\.,\-\&]+(?:S\.A\.|S\.R\.L\.|SRL|SA|S\.A\.S|SAS|E\.I\.R\.L|EIRL|LTDA|C\.V\.|CV|\.COM|LIMITADA|EURL|AG)?)', # Empresa con siglas
        r'(?:RAZ[OÓ]N\s*SOCIAL|DENOMINACI[OÓ]N|NOMBRE)[:\s]*([A-Z][A-Za-z\s\.,\-]+[A-Za-z])' # Solo nombre sin siglas
    ]

    # Buscar cerca del CUIT
    if cuit_match:
        start_index = text.find(cuit_match.group(0))
        search_area = text[max(0, start_index - 100):min(len(text), start_index + 100)] # Reducir área de búsqueda
        
        for pattern in razon_social_patterns:
            razon_social_match = re.search(pattern, search_area, re.IGNORECASE)
            if razon_social_match:
                potential_razon_social = razon_social_match.group(1).strip()
                # Evitar capturar fechas o números puros
                if not re.match(r'^\\d{1,2}[/\\.\-]?\\d{1,2}[/\\.\-]?\\d{2,4}$|^(?:\\d+(\\.\\d+)?([,\.]\\d+)?)$|^(?:(?:Efectivo|Débito|Crédito|Cheque|Transferencia))$|^FECHA|DATE|EMISION$', potential_razon_social, re.IGNORECASE):
                    data["razon_social"] = potential_razon_social
                    logger.info(f"Razón Social encontrada (cerca de CUIT): {data['razon_social']}")
                    razon_social_found = True
                    break
            if razon_social_found:
                break

    # Si no se encontró cerca del CUIT, intentar búsqueda general por líneas
    if not razon_social_found:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Buscar "Razón Social" o "Denominación" en la misma línea y capturar el resto
            for pattern in razon_social_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match and match.group(1).strip():
                    potential_razon_social = match.group(1).strip()
                    if not re.match(r'^\\d{1,2}[/\\.\-]?\\d{1,2}[/\\.\-]?\\d{2,4}$|^(?:\\d+(\\.\\d+)?([,\.]\\d+)?)$|^(?:(?:Efectivo|Débito|Crédito|Cheque|Transferencia))$|^FECHA|DATE|EMISION$', potential_razon_social, re.IGNORECASE):
                        data["razon_social"] = potential_razon_social
                        logger.info(f"Razón Social encontrada (misma línea general): {data['razon_social']}")
                        razon_social_found = True
                        break
                if razon_social_found:
                    break
            
            if razon_social_found:
                break

            # Si no, buscar la línea siguiente si la anterior contiene la palabra clave
            if ("RAZÓN SOCIAL" in line.upper() or "RAZON SOCIAL" in line.upper() or "DENOMINACION" in line.upper() or "NOMBRE" in line.upper()) and i + 1 < len(lines):
                next_line_text = lines[i + 1].strip()
                # Validar la línea siguiente para asegurar que no sea solo números o fechas
                if next_line_text and not re.match(r'^\\d+([,\.]\\d+)?([\\.\s]*%)?$|^\\d{1,2}[/\\.\-]?\\d{1,2}[/\\.\-]?\\d{2,4}$|^(?:(?:Efectivo|Débito|Crédito|Cheque|Transferencia))$', next_line_text, re.IGNORECASE): 
                    data["razon_social"] = next_line_text
                    logger.info(f"Razón Social encontrada (línea siguiente general): {data['razon_social']}")
                    razon_social_found = True
                    break

    # Buscar fecha
    fecha_match = re.search(fecha_pattern, text, re.IGNORECASE)
    if fecha_match:
        data["fecha"] = fecha_match.group(1).strip()
        logger.info(f"Fecha encontrada: {data['fecha']}")
    
    # Buscar número de factura
    factura_match = re.search(factura_pattern, text, re.IGNORECASE)
    if factura_match:
        # Limpiar el número de factura para eliminar caracteres no deseados
        raw_numero_factura = factura_match.group(1)
        cleaned_numero_factura = re.sub(r'[^0-9A-Z\-\.—]', '', raw_numero_factura).strip()
        data["numero_factura"] = cleaned_numero_factura
        logger.info(f"Número de factura encontrado: {data['numero_factura']}")
    
    # Buscar importe total
    total_match = re.search(total_pattern, text, re.IGNORECASE)
    if total_match:
        # Tomar la última coincidencia como el total más probable
        all_total_matches = re.findall(total_pattern, text, re.IGNORECASE)
        if all_total_matches:
            # Limpiar el valor: eliminar espacios, comas, y asegurar un formato de punto decimal
            # Y eliminar caracteres no numéricos o de puntuación esperados
            raw_total = all_total_matches[-1]
            cleaned_total = re.sub(r'[^0-9\.,]', '', raw_total).replace(',', '.')
            # Validar si es un número válido antes de asignar
            if re.match(r'^\\d+(\\.\\d+)?$', cleaned_total):
                data["importe_total"] = cleaned_total
                logger.info(f"Importe total encontrado: {data['importe_total']}")
            else:
                logger.warning(f"Importe total no válido después de la limpieza: {raw_total} -> {cleaned_total}")
    
    # Buscar IVA
    iva_match = re.search(iva_pattern, text, re.IGNORECASE)
    if iva_match:
        # Tomar la última coincidencia como el IVA más probable
        all_iva_matches = re.findall(iva_pattern, text, re.IGNORECASE)
        if all_iva_matches:
            # Limpiar el valor: eliminar espacios, comas, y asegurar un formato de punto decimal
            # Y eliminar caracteres no numéricos o de puntuación esperados
            raw_iva = all_iva_matches[-1]
            cleaned_iva = re.sub(r'[^0-9\.,%]', '', raw_iva).replace(',', '.') # Incluir % para IVA
            # Validar si es un número válido antes de asignar
            if re.match(r'^\\d+(\\.\\d+)?$|^\\d+%$', cleaned_iva): # Aceptar porcentaje también
                data["iva"] = cleaned_iva
                logger.info(f"IVA encontrado: {data['iva']}")
            else:
                logger.warning(f"IVA no válido después de la limpieza: {raw_iva} -> {cleaned_iva}")
    
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
