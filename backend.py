from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
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

# Importaciones de Tesseract y Layout Parser
import pytesseract
import layoutparser as lp

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Invoice OCR API")

# Inicializar Layout Parser y Tesseract (si es necesario)
# Modelos de Layout Parser se cargan una sola vez.
# layout_model = lp.models.PaddleDetectionLayoutModel(config_path=None, model_path=None, extra_config=None, device='cpu') # Ejemplo si se usara el modelo de deteccion de paddle

# Para Tesseract, no hay una inicialización global como PaddleOCR, se usa directamente.

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
def extract_text_from_image(image: Image.Image):
    try:
        # Preprocesar la imagen (convertir a escala de grises y aplicar umbralización)
        processed_image = preprocess_image(image)
        
        # Convertir la imagen procesada a un array de NumPy (para LayoutParser si se usa)
        image_np = np.array(processed_image)

        # Opcional: Detección de diseño con Layout Parser
        # Si el documento tiene una estructura compleja, esto puede ayudar a organizar el texto.
        # Ejemplo básico: Esto requiere un modelo preentrenado o entrenar uno.
        # Por ahora, simplemente usaremos pytesseract en toda la imagen, pero la importación
        # de layoutparser ya está lista para una futura implementación.

        # Ejecutar OCR con Tesseract
        # -l spa+eng: Especifica los idiomas español e inglés.
        # --psm 6: Page Segmentation Mode (PSM) 6 asume una sola columna de texto uniforme.
        # Otros PSM: 3 (valor predeterminado, automático), 4 (bloque de texto único), 11 (palabras).
        full_text = pytesseract.image_to_string(processed_image, lang='spa+eng', config='--psm 6')
        
        logger.info(f"Texto extraído de la imagen (primeros 100 chars): {full_text[:100]}")
        return full_text
    except Exception as e:
        logger.error(f"Error en OCR con Tesseract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en OCR con Tesseract: {str(e)}")

# Nueva función para preprocesar la imagen
def preprocess_image(pil_image: Image.Image):
    """Mejora la imagen para el OCR aplicando filtros.
    """
    # Convertir a escala de grises
    if pil_image.mode != 'L':
        image_np = np.array(pil_image.convert('L'))
    else:
        image_np = np.array(pil_image)
    
    # Aplicar umbralización (Otsu's Binarization)
    _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(thresh)

# Función para extraer datos específicos del texto
def extract_invoice_data(text):
    logger.info(f"Texto recibido para extracción de datos (primeras 500 chars):\n{text[:500]}")
    logger.debug(f"Texto completo para extracción de datos:\n{text}") # Logging del texto completo
    
    # Aplicar corrección de errores comunes de OCR
    text = correct_ocr_errors(text)
    logger.debug(f"Texto después de corrección de OCR:\n{text}") # Logging del texto corregido
    
    # Inicializar diccionario de datos
    data = {
        "cuit_cuil": None,
        "fecha": None,
        "razon_social": None,
        "numero_factura": None,
        "importe_total": None,
        "iva": None,
        "line_items": [] # Nueva clave para los ítems de línea
    }
    
    # Patrones de expresiones regulares para extraer datos (mejorados)
    # CUIT/CUIL: 11 dígitos con o sin guiones, o DNI/Pasaporte, o ID Fiscal
    cuit_pattern = r'(?:CUIT|CUIL|C\.U\.I\.T\.|C\.U\.I\.L\.|ID\s*FISCAL|DNI|PASAPORTE|CIF|NIF|RUC)[:\s]*([A-Z0-9]{1,3}[-\s]?\d{6,8}[-\s]?\d{1,2}|\b\d{11}\b|[A-Z]\d{7,8}[A-Z])' # Added CIF/NIF and alphanumeric pattern for CIF/NIF
    
    # Fecha: Ampliar formatos de fecha (DD/MM/AAAA, DD-MM-AAAA, DD.MM.AAAA, AAAA-MM-DD) y permitir espacios opcionales alrededor de separadores
    fecha_pattern = r'(?:FECHA|DATE|EMISION|FECHA\s*DE\s*EMISION|FECHA\s*DE\s*EMISI[OÓ]N)[:\s]*(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4}|\d{4}[/\.\-]\d{1,2}[/\.\-]\d{1,2})'
    
    # Razón Social: Buscar cerca de CUIT/CUIL o con palabras clave comunes
    # Se buscará en un rango de líneas alrededor del CUIT/CUIL o de la palabra "RAZON SOCIAL"
    
    # Número de factura: Ampliar formatos, incluyendo prefijos y sufijos comunes
    factura_pattern = r'(?:FACTURA|FAC|FACTURA\s*N[°º]?|COMPROBANTE|REMITO|Comp\.Nro|N[°º]?\s*factura|N[ÚU]MERO\s*DE\s*FACTURA)[:\s#]*([A-Z0-9]+(?:[\s\.\-\/][A-Z0-9]+)*)'
    
    # Importe total: Más robusto, considerando diferentes escrituras de moneda y separadores
    total_pattern = r'(?:TOTAL(?=.*\s*A\s*PAGAR)?|IMPORTE\s*TOTAL|GRAN\s*TOTAL|SUBTOTAL|NETO\s*A\s*PAGAR|IMPORTE\s*NETO\s*GRAVADO)[:\s$]*([\-]?\s*\$?[\s]*\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?)'
    
    # IVA: Más robusto, considerando diferentes escrituras y separadores
    iva_pattern = r'(?:IVA|I\.V\.A\.|IMPUESTO\s*VALOR\s*AGREGADO|IMPUESTOS)[:\s$]*([\-]?\s*\$?[\s]*\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?)'
    
    # Buscar CUIT/CUIL
    cuit_match = re.search(cuit_pattern, text, re.IGNORECASE)
    if cuit_match:
        data["cuit_cuil"] = cuit_match.group(1).strip()
        logger.info(f"CUIT/CUIL encontrado: {data['cuit_cuil']}")
    else:
        logger.warning("CUIT/CUIL no encontrado.")
    
    # Buscar razón social (mejorado)
    # Se buscará en un rango de líneas alrededor del CUIT/CUIL si se encontró, o cerca de "RAZÓN SOCIAL"
    # Priorizar la búsqueda en la misma línea o en las adyacentes
    razon_social_found = False
    
    # Patrones para razón social, incluyendo errores comunes de OCR y términos legales
    # Más específicos para capturar nombres de empresas y evitar fechas/números
    # Prioritize patterns with explicit keywords and company types
    razon_social_patterns = [
        # Pattern 1: Keywords like "RAZON SOCIAL", "DENOMINACION" followed by a name and optional company suffix. Anchored to end of line.
        r'''(?:RAZ[OÓ]N\s*SOCIAL|DENOMINACI[OÓ]N|NOMBRE|EMPRESA|PROVEEDOR|CLIENTE)\s*[:\s]*([A-ZÁÉÍÓÚÑÜ0-9][A-Za-zÁÉÍÓÚÑÜ0-9\s\.,'\-\&\(\)]+(?:S\.A\.|S\.R\.L\.|S\.L\.|SRL|SA|S\.A\.S\.|SAS|E\.I\.R\.L\.|EIRL|LTDA\.|C\.V\.|S\.C\.|S\.COOP\.|COOP|A\.C\.|S\.EN\s*C\.|S\.C\.P\.|S\.A\.U\.|S\.L\.U\.|C\.B\.|SC|\.COM|\.ORG|\.NET|\.INFO|\.BIZ|\.ES|\.AR|\.MX|\.CO|\.CL|\.PE|\.EC|\.VE)?)\s*(?:\n|$)''',
        # Pattern 2: Name followed by a mandatory company suffix.
        r'''([A-ZÁÉÍÓÚÑÜ0-9][A-Za-zÁÉÍÓÚÑÜ0-9\s\.,'\-\&\(\)]+)\s+(S\.A\.|S\.R\.L\.|S\.L\.|SRL|SA|S\.A\.S\.|SAS|E\.I\.R\.L\.|EIRL|LTDA\.|C\.V\.|S\.C\.|S\.COOP\.|COOP|A\.C\.|S\.EN\s*C\.|S\.C\.P\.|S\.A\.U\.|S\.L\.U\.|C\.B\.|SC)''',
        # Pattern 3: More generic, captures capitalized words, potentially a company name, often found near top or CUIT.
        # This pattern is less specific and should be used carefully, possibly with more context or post-validation.
        # It looks for a sequence of capitalized words, possibly including '.', ',', '&', '-', '(', ')'.
        # Avoids capturing lines that are clearly not company names (e.g. "FACTURA", "TOTAL", etc. handled by exclusion list)
        r'''^([A-ZÁÉÍÓÚÑÜ0-9][A-Za-zÁÉÍÓÚÑÜ0-9\s\.,'\-\&\(\)]+[A-Za-zÁÉÍÓÚÑÜ0-9\)])\s*(?:\n|$)'''
        # Removed r'''NOMBRE\s*DE\s*TU\s*EMPRESA\s*\n\s*(.+)''' as it is too specific.
    ]

    # Exclusion list for potential Razón Social matches
    razon_social_exclusion_pattern = r'(?i)^(?:[\d\s/\.\-]+)$|^(?:(?:Efectivo|Débito|Crédito|Cheque|Transferencia|Contado|Tarjeta))$|^FECHA(?:S)?$|DATE(?:S)?$|EMISION(?:ES)?$|TOTAL(?:ES)?$|IMPORTE(?:S)?$|COMPROBANTE(?:S)?$|FACTURA(?:S)?$|N[°º]?$|C\.U\.I\.T\.?$|CUIT$|CUIL$|NIF$|CIF$|RUC$|PEDIDO$|RECIBO$|ALBARAN$|NOTA$|ORDEN$|CLIENTE$|PROVEEDOR$|DOMICILIO$|DIRECCION$|CONCEPTO$|DESCRIPCION$|CANTIDAD$|PRECIO$|UNIDADES$|SUBTOTAL$|IVA$|IMPUESTO(?:S)?$|GASTO(?:S)?$|ENVIO$|PAGAR$|COBRAR$|VENCIMIENTO$|CODIGO$|REFERENCIA$'

    # Buscar cerca del CUIT
    if cuit_match:
        start_index = text.find(cuit_match.group(0))
        search_area = text[max(0, start_index - 150):min(len(text), start_index + 150)] # Reducir área de búsqueda un poco
        
        for pattern in razon_social_patterns:
            # For patterns that capture name and suffix separately (like pattern 2)
            is_name_suffix_pattern = (pattern == razon_social_patterns[1])

            razon_social_match = re.search(pattern, search_area, re.IGNORECASE | re.MULTILINE)
            if razon_social_match:
                if is_name_suffix_pattern:
                    # Concatenate name and suffix if captured separately
                    potential_razon_social = f"{razon_social_match.group(1).strip()} {razon_social_match.group(2).strip()}"
                else:
                    potential_razon_social = razon_social_match.group(1).strip()

                # Evitar capturar fechas o números puros o palabras clave de factura using the exclusion pattern
                if not re.match(razon_social_exclusion_pattern, potential_razon_social, re.IGNORECASE) and len(potential_razon_social) > 3 : # Added length check
                    data["razon_social"] = potential_razon_social
                    logger.info(f"Razón Social encontrada (cerca de CUIT, patrón: {pattern[:30]}...): {data['razon_social']}")
                    razon_social_found = True
                    break
            if razon_social_found:
                break
        if not razon_social_found:
            logger.info("Razón Social no encontrada cerca de CUIT. Intentando búsqueda general.")

    # Si no se encontró cerca del CUIT, intentar búsqueda general por líneas
    if not razon_social_found:
        lines = text.split('\n') # Ensure lines is defined if CUIT was not found either
        for i, line in enumerate(lines):
            # Buscar "Razón Social" o "Denominación" en la misma línea y capturar el resto
            for pattern in razon_social_patterns:
                is_name_suffix_pattern = (pattern == razon_social_patterns[1])
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if is_name_suffix_pattern:
                        potential_razon_social = f"{match.group(1).strip()} {match.group(2).strip()}"
                    else:
                        potential_razon_social = match.group(1).strip()

                    if potential_razon_social and not re.match(razon_social_exclusion_pattern, potential_razon_social, re.IGNORECASE) and len(potential_razon_social) > 3:
                        data["razon_social"] = potential_razon_social
                        logger.info(f"Razón Social encontrada (misma línea general, patrón: {pattern[:30]}...): {data['razon_social']}")
                        razon_social_found = True
                        break
                if razon_social_found:
                    break
            
            if razon_social_found:
                break

            # Si no, buscar la línea siguiente si la anterior contiene la palabra clave (e.g. "RAZON SOCIAL:" en una linea, nombre en la siguiente)
            # Keywords that might indicate the company name is on the next line
            next_line_keywords = ["RAZÓN SOCIAL", "RAZON SOCIAL", "DENOMINACION", "NOMBRE", "EMPRESA", "PROVEEDOR", "CLIENTE", "TITULAR", "SR./SRA." ]
            if any(keyword.upper() in line.upper() for keyword in next_line_keywords) and i + 1 < len(lines):
                next_line_text = lines[i + 1].strip()
                # Attempt to match the most generic pattern on the next line or just take it if it's not excluded
                generic_match_next_line = re.match(razon_social_patterns[2], next_line_text, re.IGNORECASE | re.MULTILINE)

                if generic_match_next_line:
                    potential_next_line_rs = generic_match_next_line.group(1).strip()
                else:
                    potential_next_line_rs = next_line_text

                if potential_next_line_rs and not re.match(razon_social_exclusion_pattern, potential_next_line_rs, re.IGNORECASE) and len(potential_next_line_rs) > 3 and not any(keyword.upper() in potential_next_line_rs.upper() for keyword in next_line_keywords):
                    data["razon_social"] = potential_next_line_rs
                    logger.info(f"Razón Social encontrada (línea siguiente a '{line.strip()[:50]}...'): {data['razon_social']}")
                    razon_social_found = True
                    break
        if not razon_social_found:
            logger.info("Razón Social no encontrada en búsqueda general por líneas. Intentando en primeras líneas.")

    # Simplified search for company names in the top part of the document if other methods fail
    if not razon_social_found:
        lines = text.split('\n') # Ensure lines is defined
        top_lines_text = "\n".join(lines[:15]) # Look in the first 15 lines combined
        for pattern in razon_social_patterns:
            is_name_suffix_pattern = (pattern == razon_social_patterns[1])
            # Try to find matches in the block of top lines
            # Using finditer to get all matches and then try to pick the best one (e.g. longest, or first)
            matches = list(re.finditer(pattern, top_lines_text, re.IGNORECASE | re.MULTILINE))
            if matches:
                for match in matches: # Iterate through all matches found by this pattern
                    if is_name_suffix_pattern:
                        potential_razon_social = f"{match.group(1).strip()} {match.group(2).strip()}"
                    else:
                        potential_razon_social = match.group(1).strip()

                    if potential_razon_social and not re.match(razon_social_exclusion_pattern, potential_razon_social, re.IGNORECASE) and len(potential_razon_social) > 3:
                        # Heuristic: prefer longer names or names with company suffixes
                        if not data["razon_social"] or len(potential_razon_social) > len(data["razon_social"]) or any(suffix in potential_razon_social.upper() for suffix in ["S.A", "S.R.L", "S.L", "LLC", "INC", "LTD"]):
                            data["razon_social"] = potential_razon_social
                            logger.info(f"Razón Social encontrada (primeras líneas, pattern: {pattern[:30]}...): {data['razon_social']}")
                            # Potentially set razon_social_found = True here if we are confident,
                            # or let it try other patterns/matches in top lines.
                            # For now, let's be greedy and take the first good one from top lines.
                            razon_social_found = True
                            break # Break from matches loop
            if razon_social_found: # This break is for the inner loop of matches from finditer
                break # Break from patterns loop if found in top lines
        if not razon_social_found:
            logger.warning("Razón Social no encontrada después de todos los métodos.")

    # Buscar fecha
    fecha_match = re.search(fecha_pattern, text, re.IGNORECASE)
    if fecha_match:
        data["fecha"] = fecha_match.group(1).strip()
        logger.info(f"Fecha encontrada: {data['fecha']}")
    else:
        logger.warning("Fecha no encontrada.")
    
    # Buscar número de factura
    factura_match = re.search(factura_pattern, text, re.IGNORECASE)
    if factura_match:
        # Limpiar el número de factura para eliminar caracteres no deseados
        raw_numero_factura = factura_match.group(1)
        # Keep '/' in the cleaned numero_factura as the regex allows it.
        cleaned_numero_factura = re.sub(r'[^0-9A-Z\-\—\/]', '', raw_numero_factura).strip()
        data["numero_factura"] = cleaned_numero_factura
        logger.info(f"Número de factura encontrado: {data['numero_factura']}")
    else:
        logger.warning("Número de factura no encontrado.")
    
    # Buscar importe total
    total_match = re.search(total_pattern, text, re.IGNORECASE)
    if total_match:
        # Tomar la última coincidencia como el total más probable
        all_total_matches = re.findall(total_pattern, text, re.IGNORECASE)
        if all_total_matches:
            raw_total = all_total_matches[-1]
            data["importe_total"] = clean_currency(raw_total)
            logger.info(f"Importe total encontrado: {data['importe_total']}")
            if data["importe_total"] is None:
                logger.warning(f"Importe total no válido después de la limpieza: {raw_total}")
    
    # Buscar IVA
    iva_match = re.search(iva_pattern, text, re.IGNORECASE)
    if iva_match:
        # Tomar la última coincidencia como el IVA más probable
        all_iva_matches = re.findall(iva_pattern, text, re.IGNORECASE)
        if all_iva_matches:
            raw_iva = all_iva_matches[-1]
            data["iva"] = clean_currency(raw_iva)
            logger.info(f"IVA encontrado: {data['iva']}")
            if data["iva"] is None:
                logger.warning(f"IVA no válido después de la limpieza: {raw_iva}")

    logger.info(f"Resumen de datos de factura extraídos: CUIT={data['cuit_cuil']}, Fecha={data['fecha']}, RazonSocial={data['razon_social']}, FacturaNro={data['numero_factura']}, Total={data['importe_total']}, IVA={data['iva']}")
    
    # Extraer ítems de línea
    data["line_items"] = extract_line_items(text)
    
    return data

# Nueva función para extraer los ítems de línea de una factura
def extract_line_items(text: str) -> list:
    items = []
    logger.info("Iniciando extracción de ítems de línea...")
    
    lines = text.split('\n')
    
    # Patrones de palabras clave para identificar el inicio y fin de la tabla de ítems.
    header_keywords = [
        "codigo", "code", "producto", "item", "descripcion", "description", "descrip", "desc.",
        "cantidad", "cant.", "qty", "unid", "unidades",
        "precio", "p.u.", "p/u", "precio unitario", "unit price", "valor unit.",
        "importe", "total", "subtotal", "valor", "sub-total", "sub total"
    ]
    # Pattern to detect a line that likely contains multiple header keywords
    # This looks for a line with at least 2-3 header-like words.
    # We compile this pattern for efficiency as it's used in a loop.
    likely_header_line_pattern = re.compile(
        r"|".join([r"\b" + kw + r"\b" for kw in header_keywords[:12]]), # Check for first few common ones
        re.IGNORECASE
    )

    # Define sub-patterns for line items
    # Optional Code/SKU: Allows alphanumeric, hyphens, dots, slashes. Requires at least 2 chars.
    code_pattern_str = r'(?:([A-Z0-9\-\./_]{2,})\s+)?'
    # Description: Non-greedy, captures most characters. At least 3 chars.
    # It should not be too greedy to capture numbers from next columns.
    # We try to stop it before a sequence of typical numeric values.
    desc_pattern_str = r'(.+?)\s+'
    # Numerical fields: Allow digits, spaces, commas, dots. Negative numbers allowed for totals.
    # Needs careful post-processing to convert to float.
    # This pattern tries to capture a number that might have thousands separators and a decimal part.
    # It expects a digit at the end of the number part.
    num_pattern_str = r'(-?[\d\s\.,]*\d)' # Basic number pattern
    
    # Combined Line Item Patterns:
    # Order of patterns matters: from more specific (more columns) to less specific.
    line_item_patterns = [
        # 1. Code (opt), Desc, Qty, Unit Price, Line Total
        re.compile(r"^\s*" + code_pattern_str + desc_pattern_str + num_pattern_str + r"\s+" + num_pattern_str + r"\s+" + num_pattern_str + r"\s*$", re.IGNORECASE),
        # 2. Code (opt), Desc, Qty, Total (Unit Price is missing)
        re.compile(r"^\s*" + code_pattern_str + desc_pattern_str + num_pattern_str + r"\s+" + num_pattern_str + r"\s*$", re.IGNORECASE),
        # 3. Code (opt), Desc, Line Total (Qty and Unit Price are missing)
        re.compile(r"^\s*" + code_pattern_str + desc_pattern_str + num_pattern_str + r"\s*$", re.IGNORECASE),
        # 4. Desc, Qty, Unit Price, Line Total (No code)
        re.compile(r"^\s*" + desc_pattern_str + num_pattern_str + r"\s+" + num_pattern_str + r"\s+" + num_pattern_str + r"\s*$", re.IGNORECASE),
        # 5. Desc, Qty, Line Total (No code, Unit Price is missing)
        re.compile(r"^\s*" + desc_pattern_str + num_pattern_str + r"\s+" + num_pattern_str + r"\s*$", re.IGNORECASE),
        # 6. Desc, Line Total (No code, Qty and Unit Price are missing)
        re.compile(r"^\s*" + desc_pattern_str + num_pattern_str + r"\s*$", re.IGNORECASE),
    ]

    # Keywords that usually signify the end of the items table or start of summary.
    table_end_keywords = [
        "subtotal", "sub-total", "sub total", "total", "importe total", "total factura",
        "descuento", "bonificacion", "iva", "i.v.a", "impuesto", "retencion", "percepcion",
        "base imponible", "neto", "total a pagar", "efectivo", "tarjeta"
    ]
    table_end_pattern = re.compile(r"|".join([r"\b" + kw + r"\b" for kw in table_end_keywords]), re.IGNORECASE)

    # Keywords for validating and cleaning description
    description_exclusion_keywords = [
        "subtotal", "total", "iva", "impuesto", "descuento", "envio", "gastos", "retencion",
        "percepcion", "importe", "valor", "neto", "bruto", "base", "cantidad", "precio",
        "codigo", "producto", "item", "descripcion", "fecha", "factura", "pedido"
    ] # Add more as needed

    def clean_number(s: str) -> Union[float, None]:
        if s is None or not isinstance(s, str):
            return None
        s = s.strip()
        if not s:
            return None

        # Remove spaces
        s_cleaned = s.replace(' ', '')

        # Standardize decimal and thousands separators
        # Count dots and commas
        num_dots = s_cleaned.count('.')
        num_commas = s_cleaned.count(',')

        if num_dots > 1 and num_commas == 1: # e.g., 1.234.567,89
            s_cleaned = s_cleaned.replace('.', '').replace(',', '.')
        elif num_commas > 1 and num_dots == 1: # e.g., 1,234,567.89
            s_cleaned = s_cleaned.replace(',', '')
        elif num_commas == 1 and num_dots == 0: # e.g., 1234,56
             s_cleaned = s_cleaned.replace(',', '.')
        # If num_dots == 1 and num_commas == 0 (e.g. 1234.56) or no separators, it's likely fine.
        # If multiple dots and multiple commas, or other complex cases, this might need more advanced logic.
        # For now, a simple removal of non-numeric except the last separator if it's a decimal.

        # Fallback: remove all non-numeric characters except the last dot or comma and hyphen
        s_cleaned_final = re.sub(r"[^0-9\.\-]", "", s_cleaned)
        if s_cleaned_final.count('.') > 1: # If multiple dots remain, remove all but last
            parts = s_cleaned_final.split('.')
            s_cleaned_final = "".join(parts[:-1]) + "." + parts[-1]

        try:
            return float(s_cleaned_final)
        except ValueError:
            # Try removing all but digits and one dot
            s_alternative = re.sub(r"[^0-9\.]", "", s_cleaned)
            if s_alternative.count('.') > 1:
                 parts = s_alternative.split('.')
                 s_alternative = "".join(parts[:-1]) + "." + parts[-1]
            try:
                return float(s_alternative)
            except ValueError:
                logger.warning(f"Could not convert number: '{s}' to float. Cleaned: '{s_cleaned_final}'")
                return None

    table_started = False
    consecutive_non_item_lines = 0
    max_consecutive_non_items = 4 # Stop if we see this many non-item lines after table start

    for line_idx, line_content in enumerate(lines):
        line = line_content.strip()
        if not line:
            continue

        # 1. Detect table header
        if not table_started:
            # A simple check: if a line contains 2 or more header keywords, consider it a header.
            if len(likely_header_line_pattern.findall(line)) >= 2: # Check if enough keywords from header_keywords are in the line
                table_started = True
                logger.info(f"Tabla de ítems INICIADA (cabecera detectada en línea {line_idx + 1}): {line}")
                continue

        if table_started:
            logger.debug(f"Procesando línea {line_idx + 1}/{len(lines)} para ítems: '{line}'")
            # 2. Check for table end conditions
            if table_end_pattern.search(line):
                # Further check if the line looks like a summary line rather than an item description
                # For example, if it contains "Total" and also a number, it's likely a summary.
                if re.search(num_pattern_str, line): # Check if there's a number in the line as well
                    logger.info(f"Fin de tabla de ítems detectado por palabra clave y número (línea {line_idx + 1}): {line}")
                    break
                else:
                    logger.info(f"Potencial fin de tabla (solo palabra clave en línea {line_idx + 1}, continuando por si acaso): {line}")


            item_found_on_line = False
            for pattern_idx, current_pattern in enumerate(line_item_patterns):
                match = current_pattern.match(line)
                if match:
                    item_data = {"code": None, "description": None, "quantity": None, "unit_price": None, "line_total": None}
                    groups = match.groups()
                    
                    # Determine pattern type based on regex and number of groups.
                    # Pattern types:
                    # A: code, desc, qty, price, total (5 groups if code, 4 if no code)
                    # B: code, desc, qty, total (4 groups if code, 3 if no code)
                    # C: code, desc, total (3 groups if code, 2 if no code)

                    desc_val = None
                    qty_val_str = None
                    price_val_str = None
                    total_val_str = None

                    if pattern_idx == 0: # Code, Desc, Qty, Price, Total
                        item_data["code"] = groups[0].strip() if groups[0] else None
                        desc_val = groups[1].strip() if groups[1] else None
                        qty_val_str = groups[2]
                        price_val_str = groups[3]
                        total_val_str = groups[4]
                    elif pattern_idx == 1: # Code, Desc, Qty, Total
                        item_data["code"] = groups[0].strip() if groups[0] else None
                        desc_val = groups[1].strip() if groups[1] else None
                        qty_val_str = groups[2]
                        total_val_str = groups[3]
                    elif pattern_idx == 2: # Code, Desc, Total
                        item_data["code"] = groups[0].strip() if groups[0] else None
                        desc_val = groups[1].strip() if groups[1] else None
                        total_val_str = groups[2]
                    elif pattern_idx == 3: # Desc, Qty, Price, Total (no code implicitly)
                        desc_val = groups[0].strip() if groups[0] else None
                        qty_val_str = groups[1]
                        price_val_str = groups[2]
                        total_val_str = groups[3]
                    elif pattern_idx == 4: # Desc, Qty, Total (no code implicitly)
                        desc_val = groups[0].strip() if groups[0] else None
                        qty_val_str = groups[1]
                        total_val_str = groups[2]
                    elif pattern_idx == 5: # Desc, Total (no code implicitly)
                        desc_val = groups[0].strip() if groups[0] else None
                        total_val_str = groups[1]

                    item_data["description"] = desc_val
                    item_data["quantity"] = clean_number(qty_val_str)
                    item_data["unit_price"] = clean_number(price_val_str)
                    item_data["line_total"] = clean_number(total_val_str)

                    # Validation:
                    # 1. Description must exist and not be a keyword itself or purely numeric
                    if not desc_val or desc_val.isdigit() or len(desc_val) < 2:
                        logger.debug(f"Línea rechazada (descripción inválida): {line} -> {desc_val}")
                        continue # Try next pattern or line

                    is_desc_excluded_keyword = False
                    for kw in description_exclusion_keywords:
                        if kw in desc_val.lower().split(): # Check if desc is just a keyword
                             # Check if the keyword is a substantial part of the description
                            if len(desc_val) < len(kw) + 5: # e.g. "Total" vs "Total amount for product X"
                                is_desc_excluded_keyword = True
                                break
                    if is_desc_excluded_keyword:
                        logger.debug(f"Línea rechazada (descripción es palabra clave): {line} -> {desc_val}")
                        continue


                    # 2. At least line_total must be a valid number.
                    if item_data["line_total"] is None:
                        logger.debug(f"Línea rechazada (total de línea no es número): {line}")
                        continue # Try next pattern

                    # 3. If quantity and unit price are present, they should multiply to approx line_total (optional heuristic)
                    # Example: if abs(qty * price - total) / total > 0.1 (10% tolerance) then warn.
                    # For now, this is skipped for simplicity.

                    items.append(item_data)
                    logger.info(f"Item de línea extraído ({current_pattern.pattern[:30]}...): {item_data}")
                    item_found_on_line = True
                    consecutive_non_item_lines = 0 # Reset counter
                    break # Found a match for this line, move to next line
            
            if not item_found_on_line:
                consecutive_non_item_lines += 1
                logger.debug(f"Línea {line_idx+1} no coincide con ningún patrón de ítem ({consecutive_non_item_lines}/{max_consecutive_non_items}): {line}")
                if consecutive_non_item_lines >= max_consecutive_non_items:
                    logger.info(f"Fin de tabla de ítems detectado (demasiadas líneas consecutivas sin ítems en línea {line_idx + 1}): {line}")
                    break

        # Fallback stop: if line is way down the document and table hasn't started, stop.
        if not table_started and line_idx > len(lines) * 0.75 : # if 3/4 down and no table
            logger.info(f"Deteniendo búsqueda de ítems (muy abajo en el documento, línea {line_idx+1}, sin inicio de tabla).")
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

# Nueva función para limpiar y convertir valores de moneda
def clean_currency(value) -> Union[float, None]:
    """Limpia una cadena de valor monetario y la convierte a float."""
    if value is None:
        return None
    try:
        # Eliminar espacios, símbolos de moneda y reemplazar comas por puntos
        cleaned_value = str(value).replace(' ', '').replace('$', '').replace('€', '').replace('£', '').replace(',', '.')
        # Eliminar cualquier guion que no sea el de un número negativo
        if cleaned_value.count('-') > 1: # Si hay más de un guion, es probable que sea un separador mal reconocido
            cleaned_value = cleaned_value.replace('-', '', 1) # Eliminar solo el primero si hay más de uno
        
        # Asegurarse de que solo el último punto sea el decimal, si hay varios
        parts = cleaned_value.split('.')
        if len(parts) > 2:
            cleaned_value = ''.join(parts[:-1]) + '.' + parts[-1]
        elif len(parts) == 2 and len(parts[-1]) != 2: # Si solo hay un punto pero no dos decimales, puede ser un separador de miles
            # Esto es una heurística y puede fallar. Asume que si no hay 2 decimales, el punto es de miles.
            cleaned_value = cleaned_value.replace('.', '')
            # Si el último segmento tiene 2 dígitos, es un decimal. Ej: 1.234,56 -> 1234.56, 1.23 -> 1.23 (no cambia)
            if len(parts[-1]) == 2:
                cleaned_value = ''.join(parts[:-1]) + '.' + parts[-1]
            else:
                cleaned_value = ''.join(parts)

        return float(Decimal(cleaned_value)) # Convertir a Decimal para mayor precisión y luego a float
    except InvalidOperation:
        logger.warning(f"InvalidOperation al limpiar valor monetario: {value}")
        return None
    except Exception as e:
        logger.error(f"Error al limpiar valor monetario '{value}': {str(e)}")
        return None

# Nueva función para corregir errores comunes de OCR
def correct_ocr_errors(text):
    """Reemplaza errores comunes de OCR en el texto."""
    replacements = {
        'O': '0',
        'I': '1',
        'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8', # Añadido por posible confusión con 8
        'g': '9', # Añadido por posible confusión con 9
        'E': '6', # Añadido por posible confusión con 6
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# End of file - small change to force a new build
