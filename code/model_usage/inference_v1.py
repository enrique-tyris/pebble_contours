import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import random
import time  # Añadimos import para medir tiempos

# Rutas
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'pebble_seg_model.pt')
test_images_path = '/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/test'
output_path = '/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/results'

# FILTRO DE CONFIANZA MÍNIMA
MIN_CONFIDENCE = 0.8  # Ajusta este valor entre 0.0 y 1.0

# Crear directorio de salida
os.makedirs(output_path, exist_ok=True)

# Cargar modelo
print("Cargando modelo...")
start_load = time.time()
model = YOLO(model_path)
model.to('cpu')  # Forzar uso de CPU
end_load = time.time()
print(f"Tiempo de carga del modelo: {end_load - start_load:.2f} segundos")

# Extraer nombre del modelo del path
model_name = Path(model_path).parent.parent.name  # Extrae 'pebbles_v2_super_filtered_HIGH_AUG'

# Obtener lista de imágenes
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = []
for ext in image_extensions:
    image_files.extend(Path(test_images_path).glob(f'*{ext}'))
    image_files.extend(Path(test_images_path).glob(f'*{ext.upper()}'))

print(f"Encontradas {len(image_files)} imágenes para procesar")

def overlay_masks_on_image(image, results, confidence_threshold=MIN_CONFIDENCE):
    """
    Superpone las máscaras de segmentación sobre la imagen original y dibuja sus centroides
    Solo muestra detecciones con confianza >= confidence_threshold
    """
    # Siempre crear una copia de la imagen original
    img = results[0].orig_img.copy()
    
    detections_shown = 0
    total_detections = 0
    
    # Procesar resultados y aplicar la máscara solo si hay detecciones
    if hasattr(results[0], 'masks') and results[0].masks is not None and hasattr(results[0].masks, 'xy'):
        boxes = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
        total_detections = len(results[0].masks.xy)
        
        for r in results:
            if not hasattr(r, 'masks') or r.masks is None:
                continue
                
            for ci, c in enumerate(r):
                if not hasattr(c, 'masks') or c.masks is None or not hasattr(c.masks, 'xy'):
                    continue
                    
                # Verificar confianza si hay cajas disponibles
                if len(boxes) > ci and len(boxes[ci]) > 4:
                    confidence = boxes[ci][4]
                    if confidence < confidence_threshold:
                        continue  # FILTRAR POR CONFIANZA
                else:
                    confidence = 1.0  # Si no hay información de confianza, asumir alta
                
                detections_shown += 1
                
                # Colores súper vivos y saturados predefinidos
                vivid_colors = [
                    [255, 0, 0],      # Rojo puro
                    [0, 255, 0],      # Verde puro  
                    [0, 0, 255],      # Azul puro
                    [255, 255, 0],    # Amarillo puro
                    [255, 0, 255],    # Magenta puro
                    [0, 255, 255],    # Cian puro
                    [255, 128, 0],    # Naranja vivo
                    [128, 0, 255],    # Violeta vivo
                    [255, 0, 128],    # Rosa vivo
                    [0, 255, 128],    # Verde lima
                    [128, 255, 0],    # Verde amarillento
                    [255, 64, 64],    # Rojo coral
                ]
                
                # Seleccionar color aleatorio de la paleta viva
                color = random.choice(vivid_colors)
                
                b_mask = np.zeros(img.shape[:2], np.uint8)
                try:
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                    
                    # Calcular centroide
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        # Si no se puede calcular el centroide, usar el centro del bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        cx = x + w//2
                        cy = y + h//2
                    
                    # Crear overlay de color
                    overlay = np.zeros_like(img, dtype=np.uint8)
                    overlay[:, :] = color
                    alpha = 0.6  # Más transparencia para colores más intensos
                    overlay_masked = cv2.bitwise_and(overlay, overlay, mask=b_mask)
                    img = cv2.addWeighted(img, 1, overlay_masked, alpha, 0)
                    
                    # Dibujar contorno más fino
                    cv2.drawContours(img, [contour], -1, color, 1)
                    
                    # Dibujar centroide
                    cv2.circle(img, (cx, cy), 3, color, -1)  # Punto sólido
                    cv2.circle(img, (cx, cy), 5, (0,0,0), 1)   # Círculo alrededor en negro
                    
                    # Agregar solo las coordenadas del centroide
                    if len(boxes) > ci:
                        cv2.putText(img, f'({cx},{cy})', 
                                   (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                    
                except (IndexError, AttributeError):
                    continue
    
    return img, detections_shown, total_detections

def process_images():
    """
    Procesa todas las imágenes y guarda los resultados
    """
    total_images_processed = 0
    total_detections_all = 0
    total_detections_filtered = 0
    total_inference_time = 0
    total_processing_time = 0
    
    for i, image_path in enumerate(image_files):
        print(f"Procesando {i+1}/{len(image_files)}: {image_path.name}")
        
        start_total = time.time()
        
        # Leer imagen
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: No se pudo leer {image_path}")
            continue
            
        # Hacer predicción
        start_inference = time.time()
        results = model.predict(image, imgsz=640, conf=0.5, verbose=False, device='cpu')
        end_inference = time.time()
        inference_time = end_inference - start_inference
        total_inference_time += inference_time
        
        # Superponer máscaras con filtro de confianza
        result_image, detections_shown, total_detections = overlay_masks_on_image(image, results)
        
        # Crear nombre de archivo con modelo y confianza
        confidence_str = f"conf{MIN_CONFIDENCE:.1f}".replace(".", "")  # conf08 para 0.8
        output_filename = f"result_{image_path.stem}_{model_name}_{confidence_str}.jpg"
        output_file = Path(output_path) / output_filename
        cv2.imwrite(str(output_file), result_image)
        
        # Actualizar estadísticas
        total_images_processed += 1
        total_detections_all += total_detections
        total_detections_filtered += detections_shown
        
        end_total = time.time()
        processing_time = end_total - start_total
        total_processing_time += processing_time
        
        # Mostrar estadísticas por imagen
        print(f"  - Tiempo de inferencia: {inference_time:.3f} segundos")
        print(f"  - Tiempo total de procesamiento: {processing_time:.3f} segundos")
        if total_detections > 0:
            print(f"  - Detectados: {total_detections} pebbles")
            print(f"  - Mostrados (conf ≥ {MIN_CONFIDENCE}): {detections_shown} pebbles")
            if results[0].boxes is not None:
                confidences = results[0].boxes.conf.cpu().numpy()
                max_conf = confidences.max() if len(confidences) > 0 else 0
                print(f"  - Confianza máxima: {max_conf:.3f}")
        else:
            print(f"  - No se detectaron pebbles")
    
    # Estadísticas finales
    print("\n" + "="*50)
    print("ESTADÍSTICAS FINALES:")
    print(f"Imágenes procesadas: {total_images_processed}")
    print(f"Tiempo promedio de inferencia: {total_inference_time/total_images_processed:.3f} segundos")
    print(f"Tiempo promedio de procesamiento: {total_processing_time/total_images_processed:.3f} segundos")
    print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Tiempo total de procesamiento: {total_processing_time:.2f} segundos")
    print(f"Total detecciones: {total_detections_all}")
    print(f"Detecciones mostradas (conf ≥ {MIN_CONFIDENCE}): {total_detections_filtered}")
    if total_detections_all > 0:
        filter_percentage = (total_detections_filtered / total_detections_all) * 100
        print(f"Porcentaje filtrado: {filter_percentage:.1f}%")

if __name__ == "__main__":
    print(f"Iniciando inferencia con confianza mínima: {MIN_CONFIDENCE}")
    print(f"Modelo: {model_name}")
    print(f"Imágenes: {test_images_path}")
    print(f"Resultados: {output_path}")
    print("-" * 50)
    
    # Procesar todas las imágenes
    process_images()
    
    print(f"\nResultados guardados en: {output_path}")
    print("\n¡Inferencia completada!")
    print(f"Revisa los resultados en: {output_path}")
