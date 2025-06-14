from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response  # Eliminado StreamingResponse
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
import cv2
import numpy as np
from decimal import Decimal, InvalidOperation
from typing import Union
import base64  # Importar base64

from mistralai import Mistral  # Importar el cliente de Mistral AI

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar la clave de API de Mistral AI
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    logger.error("La variable de entorno MISTRAL_API_KEY no está configurada.")
    # Considerar lanzar una excepción o manejar este caso según la política de errores
    # Por ahora, se dejará que la API de Mistral lance su propio error si la clave falta.

app = FastAPI(title="Invoice OCR API")

# Inicializar Layout Parser y Tesseract (si es necesario)
# Modelos de Layout Parser se cargan una sola vez.
# layout_model = lp.models.PaddleDetectionLayoutModel(config_path=None, model_path=None, extra_config=None, device='cpu')  # Ejemplo si se usara el modelo de deteccion de paddle

# Para Tesseract, no hay una inicialización global como PaddleOCR, se usa directamente.
# Se deshabilitará la inicialización global si se usa Mistral OCR.

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

# Funciones de utilidad para limpieza de datos y preprocesamiento de imagen
def clean_currency(value) -> Union[float, None]:
    """Limpia una cadena de valor monetario y la convierte a float."""
    if value is None:
        return None
    try:
        # Eliminar espacios, símbolos de moneda y reemplazar comas por puntos
        cleaned_value = str(value).replace(' ', '').replace('€', '').replace('£', '') \
            .replace(',', '.')
        # Eliminar cualquier guion que no sea el de un número negativo
        if cleaned_value.count('-') > 1:  # Si hay más de un guion, es probable que sea un separador mal reconocido
            cleaned_value = cleaned_value.replace('-', '', 1)  # Eliminar solo el primero si hay más de uno

        # Asegurarse de que solo el último punto sea el decimal, si hay varios
        parts = cleaned_value.split('.')
        if len(parts) > 2:
            cleaned_value = ''.join(parts[:-1]) + '.' + parts[-1]
        elif len(parts) == 2 and len(parts[-1]) != 2:  # Si solo hay un punto pero no dos decimales, puede ser un separador de miles
            # Esto es una heurística y puede fallar. Asume que si no hay 2 decimales, el punto es de miles.
            cleaned_value = cleaned_value.replace('.', '')
            if len(parts[-1]) == 2:
                cleaned_value = ''.join(parts[:-1]) + '.' + parts[-1]
            else:
                cleaned_value = ''.join(parts)

        return float(Decimal(cleaned_value))  # Convertir a Decimal para mayor precisión y luego a float
    except InvalidOperation:
        logger.warning(f"InvalidOperation al limpiar valor monetario: {value}")
        return None
    except Exception as e:  # Variable e debe ser utilizada aquí para el logging
        logger.error(f"Error al limpiar valor monetario '{value}': {str(e)}")
        return None

def correct_ocr_errors(text):
    """Reemplaza errores comunes de OCR en el texto."""
    replacements = {
        'O': '0',
        'I': '1',
        'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8',  # Añadido por posible confusión con 8
        'g': '9',  # Añadido por posible confusión con 9
        'E': '6',  # Añadido por posible confusión con 6
        '€': '',   # Eliminar símbolo de euro
        '£': '',   # Eliminar símbolo de libra
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def clean_number(value) -> Union[float, None]:
    """Limpia una cadena de valor numérico y la convierte a float."""
    if value is None:
        return None
    try:
        # Eliminar espacios y reemplazar comas por puntos
        cleaned_value = str(value).replace(' ', '').replace(',', '.')
        # Eliminar cualquier carácter no numérico que no sea un punto o un guion inicial
        cleaned_value = re.sub(r'[^\\d.-]', '', cleaned_value)
        # Asegurarse de que solo el primer guion sea el de un número negativo
        if cleaned_value.count('-') > 1:
            cleaned_value = cleaned_value.replace('-', '', 1)

        # Asegurarse de que solo el último punto sea el decimal
        parts = cleaned_value.split('.')
        if len(parts) > 2:
            cleaned_value = ''.join(parts[:-1]) + '.' + parts[-1]

        return float(Decimal(cleaned_value))
    except InvalidOperation:
        logger.warning(f"InvalidOperation al limpiar valor numérico: {value}")
        return None
    except Exception as e:  # Variable e debe ser utilizada aquí para el logging
        logger.error(f"Error al limpiar valor numérico '{value}': {str(e)}")
        return None

def preprocess_image(pil_image: Image.Image):
    """Mejora la imagen para el OCR aplicando filtros."""
    # Convertir a escala de grises
    if pil_image.mode != 'L':
        image_np = np.array(pil_image.convert('L'))
    else:
        image_np = np.array(pil_image)
    
    # Aplicar umbralización (Otsu's Binarization)
    _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(thresh)

# Función para extraer texto de una imagen usando OCR
def extract_text_from_image(image: Image.Image):
    try:
        # Inicializar el cliente de Mistral AI
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Convertir la imagen PIL a bytes y luego a Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # Usar PNG para mantener la calidad
        img_str_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Preparar el mensaje para la API de chat de Mistral con la imagen en Base64
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extrae todo el texto de esta factura. "
                                             "Si es posible, organiza la información en un formato "
                                             "estructurado como encabezado, ítems de línea y totales."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str_base64}"}}
                ]
            }
        ]

        # Hacer la llamada a la API de Mistral
        # Usaremos un modelo con capacidades de visión, como 'pixtral-12b-2409' o 'mistral-small-latest'
        # 'mistral-small-latest' es más eficiente en tokens.
        logger.info("Realizando llamada a la API de Mistral AI para OCR...")
        chat_response = client.chat.complete(
            model="mistral-small-latest",  # O 'pixtral-12b-2409' para más robustez en visión, si es preferible
            messages=messages,
            temperature=0.1,  # Baja temperatura para resultados más deterministas
        )
        
        full_text = chat_response.choices[0].message.content
        
        logger.info(f"Texto extraído por Mistral AI (primeros 500 chars): {full_text[:500]}")
        return full_text
    except Exception as e:  # Variable e debe ser utilizada aquí para el logging
        logger.error(f"Error al usar Mistral AI para OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en OCR con Mistral AI: {str(e)}")

# Patrones de expresión regular para Razón Social y exclusiones (GLOBALES)
razon_social_patterns = [
    r'(?:RAZ[OÓ]N\\s*SOCIAL|DENOMINACI[OÓ]N|NOMBRE|EMPRESA|PROVEEDOR|CLIENTE)\\s*[:\\s]*([A-ZÁÉÍÓÚÑÜ0-9]'
    r'[A-Za-zÁÉÍÓÚÑÜ0-9\\s\\.,\\\'\\-\\&\\(\\)]+(?:S\\.A\\.|S\\.R\\.L\\.|...))\\s*(?:\\n|$)',
    r'([A-ZÁÉÍÓÚÑÜ0-9][A-Za-zÁÉÍÓÚÑÜ0-9\\s\\.,\\\'\\-\\&\\(\\)]+)\\s+(S\\.A\\.|S\\.R\\.L\\.|...)',
    r'^(?:[A-ZÁÉÍÓÚÑÜ0-9][A-Za-zÁÉÍÓÚÑÜ0-9\\s\\.,\\\'\\-\\&\\(\\)]+[A-Za-zÁÉÍÓÚÑÜ0-9\\)])\\s*(?:\\n|$)'
]

# Exclusion list for potential Razón Social matches (GLOBAL)
razon_social_exclusion_pattern = (
    r'(?i)^(?:[\\d\\s/\\.\\-]+)$|^(?:(?:Efectivo|Débito|Crédito|Cheque|Transferencia|Contado|Tarjeta))$|'
    r'^FECHA(?:S)?$|DATE(?:S)?$|EMISION(?:ES)?$|TOTAL(?:ES)?$|IMPORTE(?:S)?$|COMPROBANTE(?:S)?$|'
    r'FACTURA(?:S)?$|N[°º]?$|C\\.U\\.I\\.T\\.?$|CUIT$|CUIL$|NIF$|CIF$|RUC$|PEDIDO$|RECIBO$|ALBARAN$|NOTA$|'
    r'ORDEN$|CLIENTE$|PROVEEDOR$|DOMICILIO$|DIRECCION$|CONCEPTO$|DESCRIPCION$|CANTIDAD$|PRECIO$|'
    r'UNIDADES$|SUBTOTAL$|IVA$|IMPUESTO(?:S)?$|GASTO(?:S)?$|ENVIO$|PAGAR$|COBRAR$|VENCIMIENTO$|'
    r'CODIGO$|REFERENCIA$|CALLE$|AVENIDA$|AVDA$|NUMERO$|NRO$|CIUDAD$|PROVINCIA$|PAIS$|C\\.P\\.?$|'
    r'CODIGO\\s*POSTAL$|ORIGINAL|FACTURA\\s*DE\\s*ABONO|NOMBRE\\s*DE\\s*TU\\s*EMPRESA|NO\\s*PEDIRO'
)

# Función para extraer datos específicos del texto
def extract_invoice_data(text):
    logger.info(f"Texto recibido para extracción de datos (primeras 500 chars):\\n{text[:500]}")
    logger.debug(f"Texto completo para extracción de datos:\\n{text}")  # Logging del texto completo
    
    # NO Aplicar corrección de errores comunes de OCR al texto completo aquí. Se hará selectivamente.
    # logger.debug(f"Texto después de corrección de OCR:\\n{text}")  # Logging del texto corregido
    
    # Inicializar diccionario de datos
    data = {
        "cuit_cuil": None,
        "fecha": None,
        "razon_social": None,
        "numero_factura": None,
        "importe_total": None,
        "iva": None,
        "line_items": []  # Nueva clave para los ítems de línea
    }

    # Extraer sección de encabezado de la salida estructurada de Mistral AI
    header_section_match = re.search(r'### Encabezado\\n(.*?)(?=\\n###|\\Z)', text, re.DOTALL | re.IGNORECASE)
    header_text = ""
    if header_section_match:
        header_text = header_section_match.group(1)
        logger.info(f"Sección de encabezado extraída:\\n{header_text[:200]}")  # Log de los primeros 200 caracteres del encabezado

    # Patrones para extraer datos del encabezado estructurado
    # CUIT/CUIL
    cuit_pattern_structured = r'- \\*\\*CUIT:\\*\\* ([\\w-]+)'
    cuit_match_structured = re.search(cuit_pattern_structured, header_text, re.IGNORECASE)
    if cuit_match_structured:
        data["cuit_cuil"] = cuit_match_structured.group(1).strip()
        logger.info(f"CUIT/CUIL encontrado (estructurado): {data['cuit_cuil']}")
    else:
        # Patrón original de CUIT/CUIL (como fallback si no se encuentra en el encabezado estructurado)
        cuit_pattern_fallback = (
            r'(?:CUIT|CUIL|C\\.U\\.I\\.T\\.|C\\.U\\.I\\.L\\.|ID\\s*FISCAL|DNI|PASAPORTE|CIF|NIF|RUC|NRO\\s*'
            r'IDENTIFICACION\\s*FISCAL|REGISTRO\\s*FISCAL)[:/\\s]*([A-Z]?\\d{7,8}[A-Z]?|\\b\\d{11}\\b)'
        )
        cuit_match_fallback = re.search(cuit_pattern_fallback, text, re.IGNORECASE)
        if cuit_match_fallback:
            data["cuit_cuil"] = cuit_match_fallback.group(1).strip()
            logger.info(f"CUIT/CUIL encontrado (fallback): {data['cuit_cuil']}")
        else:
            logger.warning("CUIT/CUIL no encontrado.")

    # Fecha
    fecha_pattern_structured = (
        r'- \\*\\*(?:Fecha de Emisión|Fecha|Date|Emisión|F\\.\\s*Emisión|FECHA\\s*DE\\s*EMISI[OÓ]N):'
        r'\\*\\* (\\d{1,2}[/\\\\.\\-]\\d{1,2}[/\\\\.\\-]\\d{2,4}|\\d{4}[/\\\\.\\-]\\d{1,2}[/\\\\.\\-]\\d{1,2})'
    )
    fecha_match_structured = re.search(fecha_pattern_structured, header_text, re.IGNORECASE)
    if fecha_match_structured:
        data["fecha"] = fecha_match_structured.group(1).strip()
        logger.info(f"Fecha encontrada (estructurado): {data['fecha']}")
    else:
        # Patrón original de Fecha (como fallback)
        fecha_pattern_fallback = (
            r'(?:FECHA|DATE|EMISION|FECHA\\s*DE\\s*EMISION|F\\.\\s*EMISION|FECHA\\s*DE\\s*EMISI[OÓ]N)'
            r'[\\s:]*(\\d{1,2}[/\\\\.\\-]\\d{1,2}[/\\\\.\\-]\\d{2,4}|\\d{4}[/\\\\.\\-]\\d{1,2}[/\\\\.\\-]\\d{1,2})'
        )
        fecha_match_fallback = re.search(fecha_pattern_fallback, text, re.IGNORECASE)
        if fecha_match_fallback:
            data["fecha"] = fecha_match_fallback.group(1).strip()
            logger.info(f"Fecha encontrada (fallback): {data['fecha']}")
        else:
            logger.warning("Fecha no encontrada.")

    # Razón Social
    razon_social_pattern_structured = r'- \\*\\*(?:Razón Social|Denominación|Nombre|Empresa|Proveedor|Cliente):\\*\\* ([^\\n]+)'
    razon_social_match_structured = re.search(razon_social_pattern_structured, header_text, re.IGNORECASE)
    if razon_social_match_structured:
        potential_rs = razon_social_match_structured.group(1).strip()
        if (potential_rs and
                not re.match(razon_social_exclusion_pattern, potential_rs, re.IGNORECASE) and
                len(potential_rs) > 3):
            data["razon_social"] = potential_rs
            logger.info(f"Razón Social encontrada (estructurado): {data['razon_social']}")
    
    # Si no se encontró en el encabezado estructurado, intenta la lógica existente (prioridad cliente, emisor, etc.)
    if data["razon_social"] is None:
        lines = text.split('\\n')
        # Buscar Razón Social del Cliente (Prioridad 1: Línea siguiente a "CLIENTE")
        for i, line in enumerate(lines):
            if "CLIENTE" in line.upper() and i + 1 < len(lines):
                potential_client_rs = lines[i + 1].strip()
                if (potential_client_rs and
                        not re.match(razon_social_exclusion_pattern, potential_client_rs, re.IGNORECASE) and
                        len(potential_client_rs) > 3):
                    data["razon_social"] = potential_client_rs
                    logger.info(f"Razón Social del Cliente encontrada (línea siguiente a 'CLIENTE'): "
                                f"{data['razon_social']}")
                    break

        if data["razon_social"] is None:
            # Prioridad 2: Razón Social del Emisor (a partir de las primeras 3 líneas)
            first_few_lines = "\\n".join(lines[:3])
            for pattern in razon_social_patterns:
                issuer_razon_social_match = re.search(pattern, first_few_lines, re.IGNORECASE | re.MULTILINE)
                if issuer_razon_social_match:
                    potential_issuer_rs = issuer_razon_social_match.group(1).strip()
                    if (potential_issuer_rs and
                            not re.match(razon_social_exclusion_pattern, potential_issuer_rs, re.IGNORECASE) and
                            len(potential_issuer_rs) > 3):
                        data["razon_social"] = potential_issuer_rs
                        logger.info(f"Razón Social del Emisor encontrada (primeras líneas, "
                                    f"patrón: {pattern[:30]}...): {data['razon_social']}")
                        break
                if data["razon_social"] is not None:
                    break

        if data["razon_social"] is None:
            # Si no se encontró en las primeras líneas, intenta cerca del CUIT
            if data["cuit_cuil"] and cuit_match_structured:  # Usa cuit_match_structured si existe, si no usa el fallback
                start_index = text.find(cuit_match_structured.group(0))  # Usa el grupo 0 para la coincidencia completa
                search_area = text[max(0, start_index - 150):min(len(text), start_index + 150)]
                for pattern in razon_social_patterns:
                    is_name_suffix_pattern = (pattern == razon_social_patterns[1])
                    razon_social_match = re.search(pattern, search_area, re.IGNORECASE | re.MULTILINE)
                    if razon_social_match:
                        if is_name_suffix_pattern:
                            potential_razon_social = (f"{razon_social_match.group(1).strip()} "
                                                      f"{razon_social_match.group(2).strip()}")
                        else:
                            potential_razon_social = razon_social_match.group(1).strip()
                        if (potential_razon_social and
                                not re.match(razon_social_exclusion_pattern, potential_razon_social, re.IGNORECASE) and
                                len(potential_razon_social) > 3):
                            data["razon_social"] = potential_razon_social
                            logger.info(f"Razón Social encontrada (cerca de CUIT, "
                                        f"patrón: {pattern[:30]}...): {data['razon_social']}")
                            break
                    if data["razon_social"] is not None:
                        break
                if data["razon_social"] is None:
                    logger.info("Razón Social no encontrada cerca de CUIT. Intentando búsqueda general.")
            elif data["cuit_cuil"] and cuit_match_fallback:  # Si el CUIT se encontró con el fallback
                start_index = text.find(cuit_match_fallback.group(0))  # Usa el grupo 0 para la coincidencia completa
                search_area = text[max(0, start_index - 150):min(len(text), start_index + 150)]
                for pattern in razon_social_patterns:
                    is_name_suffix_pattern = (pattern == razon_social_patterns[1])
                    razon_social_match = re.search(pattern, search_area, re.IGNORECASE | re.MULTILINE)
                    if razon_social_match:
                        if is_name_suffix_pattern:
                            potential_razon_social = (f"{razon_social_match.group(1).strip()} "
                                                      f"{razon_social_match.group(2).strip()}")
                        else:
                            potential_razon_social = razon_social_match.group(1).strip()
                        if (potential_razon_social and
                                not re.match(razon_social_exclusion_pattern, potential_razon_social, re.IGNORECASE) and
                                len(potential_razon_social) > 3):
                            data["razon_social"] = potential_razon_social
                            logger.info(f"Razón Social encontrada (cerca de CUIT, "
                                        f"patrón: {pattern[:30]}...): {data['razon_social']}")
                            break
                    if data["razon_social"] is not None:
                        break
                if data["razon_social"] is None:
                    logger.info("Razón Social no encontrada cerca de CUIT. Intentando búsqueda general.")

        if data["razon_social"] is None:
            logger.warning("Razón Social no encontrada cerca de CUIT. Intentando búsqueda general.")
            # Prioridad 4: Búsqueda general por líneas (lógica existente)
            for i, line in enumerate(lines):
                for pattern in razon_social_patterns:
                    is_name_suffix_pattern = (pattern == razon_social_patterns[1])
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        if is_name_suffix_pattern:
                            potential_razon_social = f"{match.group(1).strip()} {match.group(2).strip()}"
                        else:
                            potential_razon_social = match.group(1).strip()
                        if (potential_razon_social and
                                not re.match(razon_social_exclusion_pattern, potential_razon_social, re.IGNORECASE) and
                                len(potential_razon_social) > 3):
                            data["razon_social"] = potential_razon_social
                            logger.info(f"Razón Social encontrada (misma línea general, "
                                        f"patrón: {pattern[:30]}...): {data['razon_social']}")
                            break
                    if data["razon_social"] is not None:
                        break
                if data["razon_social"] is not None:
                    break
            if data["razon_social"] is None:
                logger.warning("Razón Social no encontrada en búsqueda general.")

    # Número de Factura
    factura_pattern_structured = (
        r'- \\*\\*(?:Número de Comprobante|Factura|FAC|FACTURA\\s*N[°º2]?|COMPROBANTE|REMITO|Comp\\.\\s*Nro|'
        r'N[°º2]?\\s*factura|N[ÚU]MERO\\s*DE\\s*FACTURA):\\*\\* ([A-Z0-9]+(?:[\\s\\.\\-\\/][A-Z0-9]+)*)'
    )
    factura_match_structured = re.search(factura_pattern_structured, header_text, re.IGNORECASE)
    if factura_match_structured:
        data["numero_factura"] = factura_match_structured.group(1).strip()
        logger.info(f"Número de Factura encontrado (estructurado): {data['numero_factura']}")
    else:
        # Patrón original de Número de Factura (como fallback)
        factura_pattern_fallback = (
            r'(?:FACTURA|FAC|FACTURA\\s*N[°º2]?|COMPROBANTE|REMITO|Comp\\.\\s*Nro|N[°º2]?\\s*factura|'
            r'N[ÚU]MERO\\s*DE\\s*FACTURA)[:\\s#]*([A-Z0-9]+(?:[\\s\\.\\-\\/][A-Z0-9]+)*)'
        )
        factura_match_fallback = re.search(factura_pattern_fallback, text, re.IGNORECASE)
        if factura_match_fallback:
            data["numero_factura"] = factura_match_fallback.group(1).strip()
            logger.info(f"Número de Factura encontrado (fallback): {data['numero_factura']}")
        else:
            logger.warning("Número de factura no encontrado.")

    # Importe Total (mantener patrón general)
    total_pattern = (
        r'(?:TOTAL(?=.*\\s*A\\s*PAGAR)?|IMPORTE\\s*TOTAL|GRAN\\s*TOTAL|SUBTOTAL|NETO\\s*A\\s*PAGAR|'
        r'IMPORTE\\s*NETO\\s*GRAVADO|TOTAL\\s*FACTURA)[:]?[\\s]*(?:[\\$€£]\\s*)?'
        r'([-\\s]?\\s*\\d{1,3}(?:[.,\\s]\\d{3})*(?:[.,]\\d{2})?)'
    )
    total_match = re.search(total_pattern, text, re.IGNORECASE)
    if total_match:
        cleaned_total = clean_currency(correct_ocr_errors(total_match.group(1).strip()))
        if cleaned_total is not None:
            data["importe_total"] = cleaned_total
            logger.info(f"Importe Total encontrado: {data['importe_total']}")
        else:
            logger.warning(f"Importe Total encontrado, pero no pudo ser limpiado: {total_match.group(1).strip()}")
    else:
        logger.warning("Importe Total no encontrado.")
    
    # IVA (mantener patrón general)
    iva_pattern = (
        r'(?:IVA|I\\.V\\.A\\.|IMPUESTO\\s*VALOR\\s*AGREGADO|IMPUESTOS)[:\\s]*(?:[\\$€£]\\s*)?'
        r'([-\\s]?\\s*\\d{1,3}(?:[.,\\s]\\d{3})*(?:[.,]\\d{2})?)'
    )
    iva_match = re.search(iva_pattern, text, re.IGNORECASE)
    if iva_match:
        cleaned_iva = clean_currency(correct_ocr_errors(iva_match.group(1).strip()))
        if cleaned_iva is not None:
            data["iva"] = cleaned_iva
            logger.info(f"IVA encontrado: {data['iva']}")
        else:
            logger.warning(f"IVA encontrado, pero no pudo ser limpiado: {iva_match.group(1).strip()}")
    else:
        logger.warning("IVA no encontrado.")

    logger.info(f"Resumen de datos de factura extraídos: CUIT={data['cuit_cuil']}, Fecha={data['fecha']}, "
                f"RazonSocial={data['razon_social']}, FacturaNro={data['numero_factura']}, "
                f"Total={data['importe_total']}, IVA={data['iva']}")
    
    # Extraer ítems de línea
    logger.info("Iniciando extracción de ítems de línea...")
    line_items = extract_line_items(text)
    data["line_items"] = line_items
    logger.info(f"Total de ítems de línea extraídos: {len(line_items)}")
    return data

# Nueva función para extraer los ítems de línea de una factura
def extract_line_items(text: str) -> list:
    items = []
    lines = text.split('\\n')
    table_started = False
    max_consecutive_non_items = 5  # Number of non-item lines before stopping search

    # Keywords for header detection
    header_keywords = ["descripcion", "cantidad", "precio", "importe", "iva", "producto", "servicio", "total", "subtotal", "codigo"]

    def contains_enough_keywords(line_text, keywords, min_count=3):
        found_count = 0
        for kw in keywords:
            if re.search(r'\\b' + re.escape(kw) + r'\\b', line_text, re.IGNORECASE):
                found_count += 1
        return found_count >= min_count

    # For quantities (integers or floats)
    qty_pattern = r'\\b\\d+(?:[.,]\\d+)?\\b'
    # For prices/totals (currency values) - uses clean_currency logic
    price_total_pattern = r'(?:[\\$€£]\\s*)?\\d{1,3}(?:[.,\\s]\\d{3})*(?:[.,]\\d{2})?'
    # For IVA percentage
    iva_percent_pattern = r'\\d{1,2}(?:[.,]\\d+)?%'

    # Line item patterns (more robust for different column orders and non-greedy descriptions)
    # Using f-strings to embed sub-patterns correctly. Note the double curly braces for literal ones.
    line_item_patterns = [
        fr'(?i)(?P<codigo>\\b\\d{{1,5}}\\\\b)?\\s*(?P<descripcion>(?:(?!\\s*{qty_pattern}\\\\s*'
        fr'{price_total_pattern}\\\\s*{price_total_pattern}(?:\\\\s*\\\\|?\\\\s*{iva_percent_pattern})?).)+?)\\\\s+'
        fr'(?P<cantidad>{qty_pattern})\\\\s+(?P<precio_unitario>{price_total_pattern})\\\\s+'
        fr'(?P<total>{price_total_pattern})(?:\\\\s*\\\\|?\\\\s*(?P<iva_percent>{iva_percent_pattern}))?',

        fr'(?i)(?P<descripcion>(?:(?!\\s*{qty_pattern}\\\\s*{price_total_pattern}\\\\s*'
        fr'{price_total_pattern}(?:\\\\s*\\\\|?\\\\s*{iva_percent_pattern})?).)+?)\\\\s+'
        fr'(?P<cantidad>{qty_pattern})\\\\s+(?P<precio_unitario>{price_total_pattern})\\\\s+'
        fr'(?P<total>{price_total_pattern})(?:\\\\s*\\\\|?\\\\s*(?P<percent_iva>{iva_percent_pattern}))?', # Corregido typo y renombrado para consistencia

        fr'(?i)(?P<codigo>\\b\\d{{1,5}}\\\\b)?\\s*(?P<descripcion>(?:(?!\\s*{price_total_pattern}\\\\s*'
        fr'{price_total_pattern}(?:\\\\s*\\\\|?\\\\s*{iva_percent_pattern})?).)+?)\\\\s+'
        fr'(?P<precio_unitario>{price_total_pattern})\\\\s+(?P<total>{price_total_pattern})(?:\\\\s*\\\\|?\\\\s*(?P<iva_percent>{iva_percent_pattern}))?'
    ]

    consecutive_non_item_lines = 0

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Heuristic: Detect if this line looks like a potential item line (contains at least two numbers)
        # This prevents processing irrelevant lines as item lines
        numbers_in_line = re.findall(r'\\b\\d+(?:[.,]\\d+)?\\b', line)
        if len(numbers_in_line) < 2 and not table_started:  # Require at least 2 numbers for a line to be considered an item, unless table has started
            logger.debug(f"Línea {line_idx+1} no contiene suficientes números para ser un ítem: {line}")
            consecutive_non_item_lines += 1
            if consecutive_non_item_lines >= max_consecutive_non_items and table_started:
                logger.info(f"Fin de tabla de ítems detectado (demasiadas líneas consecutivas sin ítems en línea "
                            f"{line_idx + 1}): {line}")
                break
            elif consecutive_non_item_lines >= max_consecutive_non_items and not table_started:
                # If we haven't started the table yet and many non-item lines, stop looking for header too.
                if line_idx > len(lines) * 0.5:  # If more than halfway through document
                    logger.info(f"Deteniendo búsqueda de ítems y cabecera (muy abajo en el documento, línea "
                                f"{line_idx+1}, sin inicio de tabla).")
                    break
            continue


        if not table_started:
            if contains_enough_keywords(line, header_keywords, min_count=3):
                table_started = True
                logger.info(f"Cabecera de tabla detectada en línea {line_idx+1}: {line}")
                consecutive_non_item_lines = 0  # Reset counter after finding header
                continue  # Skip header line itself

        if table_started:
            item_found_on_line = False
            # Divide the line by multiple spaces to normalize column separation, then join for regex
            parts = re.split(r'\\s{2,}', line)
            processed_line = " ".join(parts).strip()

            for pattern in line_item_patterns:
                match = re.search(pattern, processed_line)
                if match:
                    item_data = {"descripcion": None, "cantidad": None,
                                 "precio_unitario": None, "line_total": None,
                                 "iva_percent": None}

                    if match.group("descripcion"):
                        item_data["descripcion"] = match.group("descripcion").strip()

                    # Apply correct_ocr_errors and clean_number/clean_currency
                    if match.group("cantidad"):
                        item_data["cantidad"] = clean_number(correct_ocr_errors(
                            match.group("cantidad").strip()))

                    if match.group("precio_unitario"):
                        item_data["precio_unitario"] = clean_currency(correct_ocr_errors(
                            match.group("precio_unitario").strip()))

                    if match.group("total"):
                        item_data["line_total"] = clean_currency(correct_ocr_errors(
                            match.group("total").strip()))

                    # Renamed to iva_percent for consistency
                    if "iva_percent" in match.groupdict() and match.group("iva_percent"):
                        item_data["iva_percent"] = clean_number(correct_ocr_errors(
                            match.group("iva_percent").strip().replace('%', '')))
                    elif "percent_iva" in match.groupdict() and match.group("percent_iva"): # For backward compatibility with previous typo
                        item_data["iva_percent"] = clean_number(correct_ocr_errors(
                            match.group("percent_iva").strip().replace('%', '')))


                    # Validation:
                    # 1. Description must exist and not be a keyword itself or purely numeric
                    if (not item_data["descripcion"] or
                            item_data["descripcion"].isdigit() or
                            len(item_data["descripcion"]) < 2):
                        logger.debug(f"Línea rechazada (descripción inválida): {line} -> "
                                     f"{item_data['descripcion']}")
                        continue  # Try next pattern or line

                    is_desc_excluded_keyword = False
                    description_exclusion_keywords = ["total", "importe", "subtotal", "iva",
                                                      "impuesto", "gasto", "envio", "pagar",
                                                      "cobrar", "vencimiento", "codigo",
                                                      "referencia", "cliente", "proveedor",
                                                      "domicilio", "direccion", "concepto",
                                                      "cantidad", "precio", "unidades",
                                                      "fecha", "date", "emision", "factura",
                                                      "comprobante", "nro", "nif", "cif",
                                                      "cuit", "cuil", "ruc", "remito",
                                                      "pedido", "recibo", "albaran", "nota",
                                                      "orden", "calle", "avenida", "avda",
                                                      "numero", "nro", "ciudad", "provincia",
                                                      "pais", "cp", "postal", "original",
                                                      "factura de abono", "nombre de tu empresa",
                                                      "no pedido", "efectivo", "debito",
                                                      "credito", "cheque", "transferencia",
                                                      "contado", "tarjeta"]
                    for kw in description_exclusion_keywords:
                        if kw in item_data["descripcion"].lower().split():  # Check if desc is just a keyword
                            # Check if the keyword is a substantial part of the description
                            if len(item_data["descripcion"]) < len(kw) + 5:  # e.g. "Total" vs "Total amount for product X"
                                is_desc_excluded_keyword = True
                                break
                    if is_desc_excluded_keyword:
                        logger.debug(f"Línea rechazada (descripción es palabra clave): {line} -> "
                                     f"{item_data['descripcion']}")
                        continue

                    # 2. At least line_total must be a valid number.
                    if item_data["line_total"] is None:
                        logger.debug(f"Línea rechazada (total de línea no es número): {line}")
                        continue  # Try next pattern

                    # 3. If quantity and unit price are present, they should multiply to approx line_total (optional heuristic)
                    # Example: if abs(qty * price - total) / total > 0.1 (10% tolerance) then warn.
                    # For now, this is skipped for simplicity.

                    items.append(item_data)
                    logger.info(f"Item de línea extraído ({pattern.pattern[:30]}...): {item_data}")
                    item_found_on_line = True
                    consecutive_non_item_lines = 0  # Reset counter
                    break  # Found a match for this line, move to next line

            if not item_found_on_line:
                consecutive_non_item_lines += 1
                logger.debug(f"Línea {line_idx+1} no coincide con ningún patrón de ítem "
                             f"({consecutive_non_item_lines}/{max_consecutive_non_items}): {line}")
                if consecutive_non_item_lines >= max_consecutive_non_items:
                    logger.info(f"Fin de tabla de ítems detectado (demasiadas líneas consecutivas sin ítems "
                                f"en línea {line_idx + 1}): {line}")
                    break

        # Fallback stop: if line is way down the document and table hasn't started, stop.
        if not table_started and line_idx > len(lines) * 0.75:  # if 3/4 down and no table
            logger.info(f"Deteniendo búsqueda de ítems (muy abajo en el documento, línea {line_idx+1}, "
                        f"sin inicio de tabla).")
            break

    logger.info(f"Total de ítems de línea extraídos: {len(items)}")
    return items

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
            raise HTTPException(status_code=400, detail="Tipo de archivo no soportado. "
                                                         "Solo se aceptan imágenes (JPEG, PNG) o PDFs.")
        
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
            
            # Crear DataFrame para datos generales de la factura
            df_invoice_summary = pd.DataFrame([invoice_data])
            
            # Crear DataFrame para los ítems de línea
            df_line_items = pd.DataFrame(invoice_data.get("line_items", []))
            
            # Crear un buffer en memoria para el Excel
            excel_buffer = io.BytesIO()
            
            # Guardar DataFrames como Excel en diferentes hojas
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_invoice_summary.to_excel(writer, index=False, sheet_name='Datos Factura')
                if not df_line_items.empty:
                    df_line_items.to_excel(writer, index=False, sheet_name='Items de Línea')
            
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
        
    except HTTPException:  # Variable e no utilizada
        # Re-lanzar excepciones HTTP para que sean manejadas por el manejador personalizado
        raise
    except Exception:  # Variable e no utilizada
        logger.error(f"Error inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

# Para probar localmente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
