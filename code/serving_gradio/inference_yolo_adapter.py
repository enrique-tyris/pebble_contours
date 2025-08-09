# inference_yolo_adapter.py

import cv2
import numpy as np
import random
import time
from supervision.detection.overlap_filter import box_non_max_suppression, mask_non_max_suppression

def realizar_inferencia(model, image, conf=0.7, device='auto'):
    """Realiza la inferencia del modelo YOLO y devuelve los resultados raw"""
    # Detectar automáticamente si hay GPU disponible
    import torch
    
    if device == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    elif device == 'gpu':
        if torch.cuda.is_available():
            device = 0
        else:
            print("⚠️  GPU no disponible, usando CPU")
            device = 'cpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        # Si se pasa un número específico de GPU
        device = device
    
    results = model.predict(image, imgsz=640, conf=conf, verbose=False, device=device)
    return results[0].orig_img.copy(), results

def procesar_mascaras_morfologicas(results, img_final):
    """Procesa las máscaras con operaciones morfológicas"""
    img_height, img_width = img_final.shape[:2]
    img_area = img_height * img_width
    
    # Calcular áreas y kernel size
    mask_areas = []
    if hasattr(results[0], 'masks') and results[0].masks is not None and hasattr(results[0].masks, 'xy'):
        for r in results:
            if not hasattr(r, 'masks') or r.masks is None:
                continue
            for ci, c in enumerate(r):
                if not hasattr(c, 'masks') or c.masks is None or not hasattr(c.masks, 'xy'):
                    continue
                try:
                    contour_orig = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    area = cv2.contourArea(contour_orig)
                    if area > 0:
                        mask_areas.append(area)
                except:
                    continue

    avg_mask_area = np.mean(mask_areas) if mask_areas else img_area * 0.01
    kernel_size_base = max(3, min(15, int(np.sqrt(avg_mask_area) / 8)))
    kernel_size = kernel_size_base if kernel_size_base % 2 == 1 else kernel_size_base + 1
    
    # Procesar cada máscara con operaciones morfológicas
    processed_masks = []
    
    for r in results:
        if not hasattr(r, 'masks') or r.masks is None:
            continue
        for ci, c in enumerate(r):
            if not hasattr(c, 'masks') or c.masks is None or not hasattr(c.masks, 'xy'):
                continue
            try:
                contour_orig = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                mask_shape = img_final.shape[:2]
                binary_mask = np.zeros(mask_shape, np.uint8)
                cv2.drawContours(binary_mask, [contour_orig], -1, 255, cv2.FILLED)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask_eroded = cv2.erode(mask_closed, kernel, iterations=2)
                mask_final = cv2.dilate(mask_eroded, kernel, iterations=2)
                
                processed_masks.append({
                    'mask_final': mask_final,
                    'mask_id': ci
                })
            except Exception as e:
                continue
    
    return kernel_size, avg_mask_area, processed_masks

def filtrar_contornos_por_area(contours, avg_mask_area, area_factor=0.1):
    """Filtra los contornos basándose en un área mínima configurable.

    area_factor (float): fracción del área promedio usada como umbral
    (por defecto 0.1 → 10 %).
    """
    min_area = max(50, avg_mask_area * area_factor)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def aplicar_smoothing(contours):
    """Aplica suavizado a los contornos"""
    smoothed_contours = []
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
        smoothed_contours.append(smooth_contour)
    
    return smoothed_contours

def dibujar_contornos(img_final, final_masks, text_scale, alpha=0.1):
    """Dibuja los contornos, centroides y etiquetas en la imagen"""
    text_thickness = max(1, int(text_scale * 2))
    centroid_radius = max(2, int(text_scale * 4))
    centroid_border = max(1, int(text_scale * 2))
    
    # Definir colores más suaves (menos intensos)
    soft_colors = [
    [180,  50,  50],  # rojo suave
    [ 50, 180,  50],  # verde suave
    [ 50,  50, 180],  # azul suave
    [180, 180,  50],  # amarillo suave
    [180,  50, 180],  # magenta suave
    [ 50, 180, 180],  # cian suave
    [180, 128,  50],  # naranja suave
    [128,  50, 180],  # violeta suave
    [180,  20, 147],  # rosa suave
    [ 50, 150,  50],  # verde lima suave
    [180,  69,  50],  # rojo-anaranjado suave
    [138,  43, 180],  # violeta oscuro suave
    [ 50, 150, 180],  # azul celeste suave
    [180, 150,  50],  # dorado suave
    [150,  20,  60],  # carmesí suave
    [127, 180,  50],  # chartreuse suave
    [180, 105, 150],  # rosa caliente suave
    [ 30, 120, 180],  # azul dodger suave
    [180, 140,  50],  # naranja brillante suave
    [150,  85, 180],  # orquídea suave
    [ 50, 180, 154],  # primavera suave
    [180,  99,  71],  # tomate suave
    [ 72,  61, 139],  # pizarra oscuro suave
    [180, 150, 150],  # rosa claro suave
    ]

    output_image = img_final.copy()
    
    # --- 1. Preparar capas de superposición para rellenos y contornos ---
    fill_overlay_layer = output_image.copy()
    contour_overlay_layer = output_image.copy()
    
    for i, mask in enumerate(final_masks):
        contour = mask['contour']
        color = soft_colors[i % len(soft_colors)]
        # Dibujar relleno en su capa
        cv2.drawContours(fill_overlay_layer, [contour], -1, color, cv2.FILLED)
        # Dibujar contorno en su capa
        cv2.drawContours(contour_overlay_layer, [contour], -1, color, 1)

    # --- 2. Mezclar las capas con la imagen original ---
    contour_alpha = 0.5 # Alpha específico para los bordes
    # Mezclar rellenos
    output_image = cv2.addWeighted(output_image, 1 - alpha, fill_overlay_layer, alpha, 0)
    # Mezclar contornos sobre la imagen ya con rellenos
    output_image = cv2.addWeighted(output_image, 1 - contour_alpha, contour_overlay_layer, contour_alpha, 0)

    # --- 3. Dibujar centroides y texto encima de todo ---
    for i, mask in enumerate(final_masks):
        contour = mask['contour']
        color = soft_colors[i % len(soft_colors)]
        cx, cy = mask['centroid']
        
        cv2.circle(output_image, (cx, cy), centroid_radius, color, -1)
        cv2.circle(output_image, (cx, cy), centroid_radius + centroid_border, (0,0,0), 1)
        cv2.putText(output_image, f'{i+1}', (cx + centroid_radius + 5, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0,0,0), text_thickness)
    
    return output_image

def obtener_centroide(contour, mask_id):
    """Obtiene el centroide de un contorno"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return {
            'contour': contour,
            'centroid': centroid,
            'mask_id': mask_id  # Mantenemos esto por compatibilidad, pero no se usará para visualización
        }
    return None

def filtrar_mascaras_superpuestas(mascaras, iou_threshold=0.5):
    """
    Aplica NMS a las máscaras usando sus bounding boxes y máscaras.
    mascaras: lista de dicts con clave 'contour'.
    Devuelve la sublista filtrada.
    """
    if not mascaras:
        return []

    # Preparar arrays para NMS
    boxes = []
    masks = []
    for m in mascaras:
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(m['contour'])
        boxes.append([x, y, x + w, y + h, 1.0])  # score fijo a 1.0
        
        # Crear máscara binaria
        mask = np.zeros((640, 640), dtype=np.uint8)  # Tamaño fijo para NMS
        cv2.drawContours(mask, [m['contour']], -1, 255, cv2.FILLED)
        masks.append(mask)

    # Convertir a arrays numpy
    boxes_np = np.array(boxes)
    masks_np = np.stack(masks)

    # Aplicar NMS
    keep_indices = mask_non_max_suppression(
        predictions=boxes_np,
        masks=masks_np,
        iou_threshold=iou_threshold,
        mask_dimension=640
    )

    # Filtrar máscaras usando los índices
    return [mascaras[i] for i in np.where(keep_indices)[0]]

def process_yolo_on_image(model, image, remove_overlapping=True, conf=0.7, device='cpu', area_factor=0.1, return_coordinates=False):
    """Función principal que coordina todo el proceso"""
    # 1. Inferencia
    img_final, results = realizar_inferencia(model, image, conf=conf, device=device)
    
    # 2. Procesamiento morfológico
    kernel_size, avg_mask_area, processed_masks = procesar_mascaras_morfologicas(results, img_final)
    
    # 3. Aplicar smoothing y procesar contornos
    all_processed_masks = []
    for mask_data in processed_masks:
        # Primero encontrar contornos
        contours_clean, _ = cv2.findContours(mask_data['mask_final'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Luego filtrar por área
        contours_filtered = filtrar_contornos_por_area(contours_clean, avg_mask_area, area_factor)

        # Finalmente aplicar smoothing
        smoothed_contours = aplicar_smoothing(contours_filtered)
        
        # Obtener centroide de cada contorno
        for contour in smoothed_contours:
            processed_contour = obtener_centroide(contour, mask_data['mask_id'])
            if processed_contour:
                all_processed_masks.append(processed_contour)

    # 4. Eliminar solapamientos si se pide
    if remove_overlapping:
        all_processed_masks = filtrar_mascaras_superpuestas(all_processed_masks,
                                                            iou_threshold=0.5)

    # 5. Dibujar resultados
    img_diagonal = np.sqrt(img_final.shape[1]**2 + img_final.shape[0]**2)
    text_scale = max(0.3, min(1.2, img_diagonal / 2000))
    img_final = dibujar_contornos(img_final, all_processed_masks, text_scale)

    # 6. Extraer coordenadas y polígonos si se solicita
    coordenadas = []
    poligonos = []
    if return_coordinates:
        coordenadas = [mask['centroid'] for mask in all_processed_masks]
        poligonos = [mask['contour'] for mask in all_processed_masks]

    return img_final, coordenadas, poligonos
