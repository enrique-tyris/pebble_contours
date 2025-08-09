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
    Procesa una sola imagen y muestra el resultado final en pantalla
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
    
    # Crear imagen final para el resultado
    start_prep = time.time()
    img_final = results[0].orig_img.copy()
    img_height, img_width = img_final.shape[:2]
    img_area = img_height * img_width
    end_prep = time.time()
    print(f"⏱️  Preparación imagen: {(end_prep - start_prep)*1000:.1f} ms")
    
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
        print(f"📏 Área promedio de máscaras: {avg_mask_area:.0f} píxeles")
    else:
        avg_mask_area = img_area * 0.01  # Fallback: 1% del área de imagen
    
    # KERNEL ADAPTATIVO para operaciones morfológicas
    # Basado en el área promedio de las máscaras
    kernel_size_base = max(3, min(15, int(np.sqrt(avg_mask_area) / 8)))
    # Asegurar que sea impar
    kernel_size = kernel_size_base if kernel_size_base % 2 == 1 else kernel_size_base + 1
    
    # ESCALA DE TEXTO ADAPTATIVA
    # Basada en el área de la imagen y tamaño promedio de máscaras
    img_diagonal = np.sqrt(img_width**2 + img_height**2)
    text_scale = max(0.3, min(1.2, img_diagonal / 2000))  # Escala entre 0.3 y 1.2
    
    # GROSOR DE TEXTO ADAPTATIVO
    text_thickness = max(1, int(text_scale * 2))
    
    print(f"📏 PARÁMETROS ADAPTATIVOS:")
    print(f"   - Imagen: {img_width}x{img_height} ({img_area:,} píxeles)")
    print(f"   - Kernel morfológico: {kernel_size}x{kernel_size}")
    print(f"   - Escala de texto: {text_scale:.2f}")
    print(f"   - Grosor de texto: {text_thickness}")
    
    # Contadores de tiempo
    time_morphology = 0
    time_simplification = 0
    time_overlap_removal = 0
    time_drawing = 0
    
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
                    start_morph = time.time()
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
                    end_morph = time.time()
                    time_morphology += (end_morph - start_morph)
                    
                    if contours_filtered:
                        # ===== APLICAR SIMPLIFICACIÓN CON APPROXPOLYDP =====
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
                    
        # ===== ELIMINAR MÁSCARAS OVERLAPPING =====
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
        
        end_overlap = time.time()
        time_overlap_removal = end_overlap - start_overlap
        
        # ===== DIBUJAR RESULTADO FINAL CON TEXTO ADAPTATIVO =====
        start_draw = time.time()
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
        end_draw = time.time()
        time_drawing = end_draw - start_draw
        
        print(f"Máscaras después del filtrado: {len(final_masks)}")
        print(f"Máscaras eliminadas por overlap: {len(masks_to_remove)}")
        
        end_processing = time.time()
        
        # Mostrar tiempos de procesamiento detallados
        print(f"\n📊 TIEMPOS DE PROCESAMIENTO ({mask_count} máscaras):")
        print(f"⏱️  Operaciones morfológicas: {time_morphology*1000:.1f} ms")
        print(f"⏱️  Simplificación (approxPolyDP): {time_simplification*1000:.1f} ms")
        print(f"⏱️  Eliminación overlapping: {time_overlap_removal*1000:.1f} ms")
        print(f"⏱️  Dibujo resultado final: {time_drawing*1000:.1f} ms")
        print(f"⏱️  Total procesamiento máscaras: {(end_processing - start_processing)*1000:.1f} ms")
        
    else:
        print("No se detectaron máscaras en la imagen")
    
    # Mostrar solo la imagen final
    start_display = time.time()
    
    # Ventana con tamaño original
    cv2.namedWindow('RESULTADO FINAL - Mascaras Procesadas', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RESULTADO FINAL - Mascaras Procesadas', img_final.shape[1], img_final.shape[0])
    cv2.imshow('RESULTADO FINAL - Mascaras Procesadas', img_final)
    
    end_display = time.time()
    
    end_total = time.time()
    
    print(f"\n📊 RESUMEN DE TIEMPOS:")
    print(f"⏱️  Carga modelo: {(end_model - start_model)*1000:.1f} ms")
    print(f"⏱️  Predicción YOLO: {(end_predict - start_predict)*1000:.1f} ms")  
    print(f"⏱️  Procesamiento total: {(end_processing - start_processing)*1000:.1f} ms")
    print(f"⏱️  Mostrar ventana: {(end_display - start_display)*1000:.1f} ms")
    print(f"⏱️  TIEMPO TOTAL: {(end_total - start_total)*1000:.1f} ms")
    
    print(f"\n✅ Resultado final:")
    print(f"   - Aplicadas operaciones morfológicas con kernel {kernel_size}x{kernel_size}")
    print(f"   - Simplificados contornos con approxPolyDP")
    print(f"   - Eliminadas máscaras superpuestas")
    print(f"   - Texto adaptativo escala {text_scale:.2f}")
    print(f"   - {len(final_masks) if 'final_masks' in locals() else 0} máscaras finales mostradas")
    print("\nPresiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_single_image()
