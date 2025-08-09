import cv2
import numpy as np
from ultralytics import YOLO
import random
import time
import os
from pathlib import Path
import csv

# Rutas
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'pebble_seg_model.pt')
images_base_path = '/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/datasets/pebbles_v4_super_filtered/images/'
output_path = '/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/results/imgs_deliverable'
csv_output_path = '/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/results/imgs_deliverable/resultados.csv'

def process_single_image(model, image_path, save_path):
    """
    Procesa una sola imagen y guarda el resultado
    Retorna el número de máscaras finales detectadas
    """
    print(f"Procesando: {image_path.name}")
    
    # Leer imagen
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: No se pudo leer {image_path}")
        return 0
        
    # Hacer predicción EN GPU
    results = model.predict(image, imgsz=640, conf=0.5, verbose=False, device=0)
    
    # Crear imagen final para el resultado
    img_final = results[0].orig_img.copy()
    img_height, img_width = img_final.shape[:2]
    img_area = img_height * img_width
    
    # ===== CÁLCULO DE PARÁMETROS ADAPTATIVOS =====
    # Calcular área promedio de máscaras para hacer kernel adaptativo
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
    
    # Parámetros adaptativos basados en imagen y máscaras
    if mask_areas:
        avg_mask_area = np.mean(mask_areas)
    else:
        avg_mask_area = img_area * 0.01  # Fallback: 1% del área de imagen
    
    # KERNEL ADAPTATIVO para operaciones morfológicas
    kernel_size_base = max(3, min(15, int(np.sqrt(avg_mask_area) / 8)))
    kernel_size = kernel_size_base if kernel_size_base % 2 == 1 else kernel_size_base + 1
    
    # ESCALA DE TEXTO ADAPTATIVA
    img_diagonal = np.sqrt(img_width**2 + img_height**2)
    text_scale = max(0.3, min(1.2, img_diagonal / 2000))
    text_thickness = max(1, int(text_scale * 2))
    
    # Lista para almacenar todas las máscaras procesadas
    all_processed_masks = []
    
    if hasattr(results[0], 'masks') and results[0].masks is not None and hasattr(results[0].masks, 'xy'):
        for r in results:
            if not hasattr(r, 'masks') or r.masks is None:
                continue
                
            for ci, c in enumerate(r):
                if not hasattr(c, 'masks') or c.masks is None or not hasattr(c.masks, 'xy'):
                    continue
                    
                # Color aleatorio vivo - PALETA AMPLIADA
                vivid_colors = [
                    [255, 0, 0],      # Rojo puro
                    [0, 255, 0],      # Verde puro  
                    [0, 0, 255],      # Azul puro
                    [255, 255, 0],    # Amarillo puro
                    [255, 0, 255],    # Magenta puro
                    [0, 255, 255],    # Cian puro
                    [255, 128, 0],    # Naranja vivo
                    [128, 0, 255],    # Violeta vivo
                    [255, 20, 147],   # Rosa intenso (Deep Pink)
                    [50, 205, 50],    # Verde lima
                    [255, 69, 0],     # Rojo naranja
                    [138, 43, 226],   # Violeta azulado
                    [0, 191, 255],    # Azul cielo intenso
                    [255, 215, 0],    # Dorado vivo
                    [220, 20, 60],    # Carmesí
                    [127, 255, 0],    # Verde chartreuse
                    [255, 105, 180],  # Rosa caliente
                    [30, 144, 255],   # Azul dodger
                    [255, 140, 0],    # Naranja oscuro
                    [186, 85, 211],   # Orquídea medio
                    [0, 250, 154],    # Verde primavera medio
                    [255, 99, 71],    # Tomate
                    [72, 61, 139],    # Azul pizarra oscuro
                    [255, 192, 203],  # Rosa claro
                ]
                color = random.choice(vivid_colors)
                
                try:
                    # Obtener contorno usando el método correcto de YOLO
                    contour_orig = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    
                    # ===== APLICAR OPERACIONES MORFOLÓGICAS CON KERNEL ADAPTATIVO =====
                    # Crear máscara binaria del contorno para aplicar morfología
                    mask_shape = img_final.shape[:2]
                    binary_mask = np.zeros(mask_shape, np.uint8)
                    cv2.drawContours(binary_mask, [contour_orig], -1, 255, cv2.FILLED)
                    
                    # Definir kernel ADAPTATIVO para operaciones morfológicas
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    
                    # Aplicar operaciones morfológicas - CONFIGURACIÓN AGRESIVA
                    # 1. Apertura: elimina pequeños ruidos y componentes desconectadas (2 iteraciones)
                    mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                    
                    # 2. Cierre: conecta pequeños huecos (2 iteraciones)
                    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
                    
                    # 3. Erosión seguida de dilatación para limpiar más (2 iteraciones cada una)
                    mask_eroded = cv2.erode(mask_closed, kernel, iterations=2)
                    mask_final = cv2.dilate(mask_eroded, kernel, iterations=2)
                    
                    # Obtener contornos limpiados
                    contours_clean, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Área mínima adaptativa (proporción del área promedio de máscaras)
                    min_area = max(50, avg_mask_area * 0.1)  # Mínimo 10% del área promedio
                    contours_filtered = [cnt for cnt in contours_clean if cv2.contourArea(cnt) > min_area]
                    
                    if contours_filtered:
                        # ===== APLICAR SIMPLIFICACIÓN CON APPROXPOLYDP =====
                        contours_simplified = []
                        for contour in contours_filtered:
                            # Calcular epsilon como 0.5% del perímetro del contorno
                            epsilon = 0.005 * cv2.arcLength(contour, True)
                            # Aplicar aproximación poligonal
                            smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
                            contours_simplified.append(smooth_contour)
                        
                        # Guardar máscaras procesadas para filtrado de overlapping
                        for contour in contours_simplified:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                                area = cv2.contourArea(contour)
                                all_processed_masks.append({
                                    'contour': contour,
                                    'centroid': centroid,
                                    'area': area,
                                    'color': color,
                                    'mask_id': ci
                                })
                    
                except (IndexError, AttributeError) as e:
                    continue
                    
        # ===== ELIMINAR MÁSCARAS OVERLAPPING =====
        # Marcar máscaras para eliminación
        masks_to_remove = set()
        
        for i in range(len(all_processed_masks)):
            if i in masks_to_remove:
                continue
                
            mask_a = all_processed_masks[i]
            centroid_a = mask_a['centroid']
            
            for j in range(i + 1, len(all_processed_masks)):
                if j in masks_to_remove:
                    continue
                    
                mask_b = all_processed_masks[j]
                contour_b = mask_b['contour']
                
                # Verificar si el centroide de A está dentro del contorno de B
                result_a_in_b = cv2.pointPolygonTest(contour_b, centroid_a, False)
                
                # Verificar si el centroide de B está dentro del contorno de A
                centroid_b = mask_b['centroid']
                contour_a = mask_a['contour']
                result_b_in_a = cv2.pointPolygonTest(contour_a, centroid_b, False)
                
                # Si hay overlap, eliminar la máscara más pequeña
                if result_a_in_b >= 0:  # A está dentro de B
                    if mask_a['area'] <= mask_b['area']:
                        masks_to_remove.add(i)
                    else:
                        masks_to_remove.add(j)
                elif result_b_in_a >= 0:  # B está dentro de A
                    if mask_b['area'] <= mask_a['area']:
                        masks_to_remove.add(j)
                    else:
                        masks_to_remove.add(i)
        
        # ===== DIBUJAR RESULTADO FINAL CON TEXTO ADAPTATIVO =====
        final_masks = [mask for i, mask in enumerate(all_processed_masks) if i not in masks_to_remove]
        
        # Radio de centroide adaptativo
        centroid_radius = max(2, int(text_scale * 4))
        centroid_border = max(1, int(text_scale * 2))
        
        for mask in final_masks:
            contour = mask['contour']
            color = mask['color']
            cx, cy = mask['centroid']
            
            # Crear máscara binaria para este contorno
            b_mask = np.zeros(img_final.shape[:2], np.uint8)
            cv2.drawContours(b_mask, [contour], -1, 255, cv2.FILLED)
            
            # Crear overlay de color relleno
            overlay = np.zeros_like(img_final, dtype=np.uint8)
            overlay[:, :] = color
            alpha = 0.6  # Transparencia para colores más intensos
            overlay_masked = cv2.bitwise_and(overlay, overlay, mask=b_mask)
            img_final = cv2.addWeighted(img_final, 1, overlay_masked, alpha, 0)
            
            # Dibujar contorno más fino encima del relleno
            cv2.drawContours(img_final, [contour], -1, color, 1)
            
            # Dibujar centroide con tamaño adaptativo
            cv2.circle(img_final, (cx, cy), centroid_radius, color, -1)  # Punto sólido
            cv2.circle(img_final, (cx, cy), centroid_radius + centroid_border, (0,0,0), 1)  # Círculo negro alrededor
            
            # Texto con escala y grosor adaptativo
            cv2.putText(img_final, f'({cx},{cy})', 
                      (cx + centroid_radius + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                      text_scale, (0,0,0), text_thickness)
        
        # Guardar imagen procesada
        cv2.imwrite(str(save_path), img_final)
        
        print(f"  ✅ Guardada: {save_path.name} - {len(final_masks)} máscaras finales")
        return len(final_masks)
        
    else:
        # No hay máscaras, guardar imagen original
        cv2.imwrite(str(save_path), img_final)
        print(f"  ✅ Guardada: {save_path.name} - 0 máscaras")
        return 0

def process_all_images():
    """
    Procesa todas las imágenes encontradas en la carpeta base y subcarpetas
    """
    start_total = time.time()
    
    # Crear carpeta de salida
    os.makedirs(output_path, exist_ok=True)
    print(f"📁 Carpeta de salida creada/verificada: {output_path}")
    
    # Cargar modelo una sola vez EN CPU
    print("🤖 Cargando modelo en GPU...")
    start_model = time.time()
    model = YOLO(model_path)
    model.to('cpu')  # Forzar uso de CPU
    end_model = time.time()
    print(f"⏱️  Modelo cargado en CPU en: {(end_model - start_model)*1000:.1f} ms")
    print(f"🚀 Dispositivo del modelo: {next(model.model.parameters()).device}")
    
    # Buscar todas las imágenes recursivamente
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_base_path).rglob(f'*{ext}'))
        image_files.extend(Path(images_base_path).rglob(f'*{ext.upper()}'))
    
    print(f"🔍 Encontradas {len(image_files)} imágenes para procesar")
    
    if len(image_files) == 0:
        print(f"❌ No se encontraron imágenes en: {images_base_path}")
        return
    
    # Crear archivo CSV
    csv_data = []
    
    # Procesar cada imagen
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Procesando: {image_path.name}")
        
        # Crear nombre de archivo de salida
        output_filename = f"processed_{image_path.stem}.jpg"
        save_path = Path(output_path) / output_filename
        
        # Procesar imagen
        try:
            num_masks = process_single_image(model, image_path, save_path)
            
            # Agregar al CSV
            csv_data.append({
                'imagen': image_path.name,
                'num_mascaras_finales': num_masks
            })
            
        except Exception as e:
            print(f"  ❌ Error procesando {image_path.name}: {e}")
            csv_data.append({
                'imagen': image_path.name,
                'num_mascaras_finales': 0
            })
    
    # Guardar CSV
    print(f"\n💾 Guardando resultados en CSV...")
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['imagen', 'num_mascaras_finales']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    end_total = time.time()
    
    # Estadísticas finales
    total_masks = sum(row['num_mascaras_finales'] for row in csv_data)
    images_with_masks = sum(1 for row in csv_data if row['num_mascaras_finales'] > 0)
    
    print(f"\n" + "="*60)
    print(f"📊 PROCESAMIENTO COMPLETADO")
    print(f"="*60)
    print(f"📂 Imágenes procesadas: {len(image_files)}")
    print(f"🎯 Imágenes con máscaras: {images_with_masks}")
    print(f"🔢 Total máscaras finales: {total_masks}")
    print(f"📈 Promedio máscaras por imagen: {total_masks/len(image_files):.2f}")
    print(f"⏱️  Tiempo total: {(end_total - start_total)/60:.2f} minutos")
    print(f"⚡ Tiempo promedio por imagen: {(end_total - start_total)/len(image_files):.2f} segundos")
    print(f"")
    print(f"📁 Imágenes guardadas en: {output_path}")
    print(f"📄 CSV guardado en: {csv_output_path}")
    print(f"")
    print(f"✅ ¡PROCESO COMPLETADO!")

if __name__ == "__main__":
    print("🚀 Iniciando procesamiento masivo de imágenes EN GPU...")
    print(f"📂 Carpeta de imágenes: {images_base_path}")
    print(f"📁 Carpeta de salida: {output_path}")
    print("-" * 60)
    
    process_all_images()
