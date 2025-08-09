import cv2
import numpy as np
from ultralytics import YOLO
import random
import time
import os

# Rutas
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'pebble_seg_model.pt')
image_path = '/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/test/pebble1.jpg'

def process_single_image():
    """
    Procesa una sola imagen y muestra el resultado en pantalla
    """
    start_total = time.time()
    
    # Cargar modelo
    print("Cargando modelo...")
    start_model = time.time()
    model = YOLO(model_path)
    model.to('cpu')
    end_model = time.time()
    print(f"⏱️  Carga del modelo: {(end_model - start_model)*1000:.1f} ms")
    
    # Leer imagen
    print(f"Procesando imagen: pebble1.jpg")
    start_load = time.time()
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo leer la imagen")
        return
    end_load = time.time()
    print(f"⏱️  Carga de imagen: {(end_load - start_load)*1000:.1f} ms")
        
    # Hacer predicción
    start_predict = time.time()
    results = model.predict(image, imgsz=640, conf=0.7, verbose=False, device='cpu')
    end_predict = time.time()
    print(f"⏱️  Predicción YOLO: {(end_predict - start_predict)*1000:.1f} ms")
    
    # Crear copias de la imagen original para las CUATRO ventanas
    start_prep = time.time()
    img_original = results[0].orig_img.copy()
    img_cleaned = results[0].orig_img.copy()
    img_simplified = results[0].orig_img.copy()
    img_filtered = results[0].orig_img.copy()
    end_prep = time.time()
    print(f"⏱️  Preparación imágenes: {(end_prep - start_prep)*1000:.1f} ms")
    
    # Contadores de tiempo
    time_morphology = 0
    time_simplification = 0
    time_drawing = 0
    time_overlap_removal = 0
    
    # Lista para almacenar todas las máscaras procesadas
    all_processed_masks = []
    
    if hasattr(results[0], 'masks') and results[0].masks is not None and hasattr(results[0].masks, 'xy'):
        print(f"Total de máscaras detectadas: {len(results[0].masks.xy)}")
        
        start_processing = time.time()
        mask_count = 0
        
        for r in results:
            if not hasattr(r, 'masks') or r.masks is None:
                continue
                
            for ci, c in enumerate(r):
                if not hasattr(c, 'masks') or c.masks is None or not hasattr(c.masks, 'xy'):
                    continue
                    
                mask_count += 1
                start_mask = time.time()
                    
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
                    
                    # ===== VENTANA 1: MÁSCARAS ORIGINALES =====
                    start_draw1 = time.time()
                    cv2.drawContours(img_original, [contour_orig], -1, color, 2)
                    
                    # Calcular centroide original
                    M = cv2.moments(contour_orig)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Dibujar centroide
                        cv2.circle(img_original, (cx, cy), 3, color, -1)
                        cv2.circle(img_original, (cx, cy), 5, (0,0,0), 1)
                        cv2.putText(img_original, f'({cx},{cy})', 
                                  (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                    end_draw1 = time.time()
                    time_drawing += (end_draw1 - start_draw1)
                    
                    # ===== VENTANA 2: MÁSCARAS LIMPIADAS CON OPERACIONES MORFOLÓGICAS =====
                    start_morph = time.time()
                    # Crear máscara binaria del contorno para aplicar morfología
                    mask_shape = img_cleaned.shape[:2]
                    binary_mask = np.zeros(mask_shape, np.uint8)
                    cv2.drawContours(binary_mask, [contour_orig], -1, 255, cv2.FILLED)
                    
                    # Definir kernel para operaciones morfológicas
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    
                    # Aplicar operaciones morfológicas - CONFIGURACIÓN MÁS AGRESIVA
                    # 1. Apertura: elimina pequeños ruidos y componentes desconectadas (2 iteraciones)
                    mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                    
                    # 2. Cierre: conecta pequeños huecos (2 iteraciones)
                    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
                    
                    # 3. Erosión seguida de dilatación para limpiar más (2 iteraciones cada una)
                    mask_eroded = cv2.erode(mask_closed, kernel, iterations=2)
                    mask_final = cv2.dilate(mask_eroded, kernel, iterations=2)
                    
                    # Obtener contornos limpiados
                    contours_clean, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filtrar contornos por área mínima (también más agresivo)
                    min_area = 100
                    contours_filtered = [cnt for cnt in contours_clean if cv2.contourArea(cnt) > min_area]
                    end_morph = time.time()
                    time_morphology += (end_morph - start_morph)
                    
                    if contours_filtered:
                        start_draw2 = time.time()
                        cv2.drawContours(img_cleaned, contours_filtered, -1, color, 2)
                        
                        # Calcular y dibujar centroide para cada contorno limpiado
                        for contour in contours_filtered:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # Dibujar centroide
                                cv2.circle(img_cleaned, (cx, cy), 3, color, -1)
                                cv2.circle(img_cleaned, (cx, cy), 5, (0,0,0), 1)
                                cv2.putText(img_cleaned, f'({cx},{cy})', 
                                          (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                        end_draw2 = time.time()
                        time_drawing += (end_draw2 - start_draw2)
                        
                        # ===== VENTANA 3: MÁSCARAS SIMPLIFICADAS CON APPROXPOLYDP =====
                        start_simp = time.time()
                        contours_simplified = []
                        for contour in contours_filtered:
                            # Calcular epsilon como 0.5% del perímetro del contorno
                            epsilon = 0.005 * cv2.arcLength(contour, True)
                            # Aplicar aproximación poligonal
                            smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
                            contours_simplified.append(smooth_contour)
                            
                            # Mostrar información de simplificación
                            print(f"Contorno {ci}: {len(contour)} puntos -> {len(smooth_contour)} puntos")
                        end_simp = time.time()
                        time_simplification += (end_simp - start_simp)
                        
                        if contours_simplified:
                            start_draw3 = time.time()
                            cv2.drawContours(img_simplified, contours_simplified, -1, color, 2)
                            
                            # Calcular y dibujar centroide para cada contorno simplificado
                            for contour in contours_simplified:
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    # Dibujar centroide
                                    cv2.circle(img_simplified, (cx, cy), 3, color, -1)
                                    cv2.circle(img_simplified, (cx, cy), 5, (0,0,0), 1)
                                    cv2.putText(img_simplified, f'({cx},{cy})', 
                                              (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                            end_draw3 = time.time()
                            time_drawing += (end_draw3 - start_draw3)
                            
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
                    print(f"Error procesando máscara {ci}: {e}")
                    continue
                    
        # ===== VENTANA 4: ELIMINAR MÁSCARAS OVERLAPPING =====
        start_overlap = time.time()
        
        print(f"\n🔍 ELIMINANDO MÁSCARAS OVERLAPPING...")
        print(f"Máscaras antes del filtrado: {len(all_processed_masks)}")
        
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
                        print(f"  Eliminando máscara {mask_a['mask_id']} (área: {mask_a['area']:.0f}) contenida en máscara {mask_b['mask_id']} (área: {mask_b['area']:.0f})")
                    else:
                        masks_to_remove.add(j)
                        print(f"  Eliminando máscara {mask_b['mask_id']} (área: {mask_b['area']:.0f}) contenida en máscara {mask_a['mask_id']} (área: {mask_a['area']:.0f})")
                elif result_b_in_a >= 0:  # B está dentro de A
                    if mask_b['area'] <= mask_a['area']:
                        masks_to_remove.add(j)
                        print(f"  Eliminando máscara {mask_b['mask_id']} (área: {mask_b['area']:.0f}) contenida en máscara {mask_a['mask_id']} (área: {mask_a['area']:.0f})")
                    else:
                        masks_to_remove.add(i)
                        print(f"  Eliminando máscara {mask_a['mask_id']} (área: {mask_a['area']:.0f}) contenida en máscara {mask_b['mask_id']} (área: {mask_b['area']:.0f})")
        
        # Dibujar máscaras finales sin overlapping
        final_masks = [mask for i, mask in enumerate(all_processed_masks) if i not in masks_to_remove]
        
        for mask in final_masks:
            cv2.drawContours(img_filtered, [mask['contour']], -1, mask['color'], 2)
            cx, cy = mask['centroid']
            cv2.circle(img_filtered, (cx, cy), 3, mask['color'], -1)
            cv2.circle(img_filtered, (cx, cy), 5, (0,0,0), 1)
            cv2.putText(img_filtered, f'({cx},{cy})', 
                      (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        
        end_overlap = time.time()
        time_overlap_removal = end_overlap - start_overlap
        
        print(f"Máscaras después del filtrado: {len(final_masks)}")
        print(f"Máscaras eliminadas por overlap: {len(masks_to_remove)}")
        
        end_processing = time.time()
        
        # Mostrar tiempos de procesamiento detallados
        print(f"\n📊 TIEMPOS DE PROCESAMIENTO ({mask_count} máscaras):")
        print(f"⏱️  Operaciones morfológicas: {time_morphology*1000:.1f} ms")
        print(f"⏱️  Simplificación (approxPolyDP): {time_simplification*1000:.1f} ms")
        print(f"⏱️  Eliminación overlapping: {time_overlap_removal*1000:.1f} ms")
        print(f"⏱️  Dibujo de contornos: {time_drawing*1000:.1f} ms")
        print(f"⏱️  Total procesamiento máscaras: {(end_processing - start_processing)*1000:.1f} ms")
        
    else:
        print("No se detectaron máscaras en la imagen")
    
    # Mostrar las CUATRO ventanas
    start_display = time.time()
    cv2.imshow('1. ANTES - Mascaras Originales', img_original)
    cv2.imshow('2. MORFOLOGIA - Mascaras Limpiadas', img_cleaned)
    cv2.imshow('3. SIMPLIFICADO - Morfologia + ApproxPolyDP', img_simplified)
    cv2.imshow('4. FILTRADO - Sin Overlapping', img_filtered)
    end_display = time.time()
    
    end_total = time.time()
    
    print(f"\n📊 RESUMEN DE TIEMPOS:")
    print(f"⏱️  Carga modelo: {(end_model - start_model)*1000:.1f} ms")
    print(f"⏱️  Predicción YOLO: {(end_predict - start_predict)*1000:.1f} ms")  
    print(f"⏱️  Procesamiento: {(end_processing - start_processing)*1000:.1f} ms")
    print(f"⏱️  Mostrar ventanas: {(end_display - start_display)*1000:.1f} ms")
    print(f"⏱️  TIEMPO TOTAL: {(end_total - start_total)*1000:.1f} ms")
    
    print("\nSe muestran cuatro ventanas:")
    print("1. ANTES - Mascaras Originales: máscaras tal como salen del modelo")
    print("2. MORFOLOGIA - Mascaras Limpiadas: después de operaciones morfológicas")
    print("3. SIMPLIFICADO - Simplificadas: morfología + approxPolyDP (menos puntos)")
    print("4. FILTRADO - Sin Overlapping: eliminadas máscaras con centros superpuestos")
    print("Presiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_single_image()