import cv2
import numpy as np
import os
import glob
import random
from pathlib import Path

def calcular_epsilon_por_porcentaje_frame(contours, img_width, img_height):
    """Calcula epsilon basado en el porcentaje del frame ocupado por los guijarros"""
    # Limites: 0.015 para 0% ocupacion, 0.005 para 10%+ ocupacion
    min_epsilon = 0.005
    max_epsilon = 0.012
    
    # Calcular area total del frame
    frame_area = img_width * img_height
    
    # Calcular area total ocupada por los guijarros
    total_pebble_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        total_pebble_area += area
    
    # Calcular porcentaje de ocupacion
    porcentaje_ocupacion = (total_pebble_area / frame_area) * 100
    
    # Distribucion uniforme entre 0% y 5%
    if porcentaje_ocupacion >= 5.0:
        normalized = 1.0
    else:
        normalized = porcentaje_ocupacion / 5.0
    
    # Calcular epsilon (inversamente proporcional al porcentaje)
    epsilon = max_epsilon - (max_epsilon - min_epsilon) * normalized
    
    return epsilon, porcentaje_ocupacion

def aplicar_smoothing(contours, img_width, img_height):
    """Aplica suavizado a los contornos usando approxPolyDP con epsilon adaptativo"""
    smoothed_contours = []
    
    # Calcular epsilon basado en el porcentaje de ocupacion del frame
    epsilon_base, porcentaje = calcular_epsilon_por_porcentaje_frame(contours, img_width, img_height)
    
    for contour in contours:
        # Ajustar epsilon por la longitud del contorno individual
        epsilon = epsilon_base * cv2.arcLength(contour, True)
        smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
        smoothed_contours.append(smooth_contour)
    
    return smoothed_contours, epsilon_base, porcentaje

def yolo_seg_to_contours(yolo_seg_line, img_width, img_height):
    """Convierte una linea de segmentacion YOLO a contornos"""
    parts = yolo_seg_line.strip().split()
    if len(parts) < 9:  # Minimo: clase + 8 coordenadas (4 puntos)
        return []
    
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    
    # Convertir coordenadas normalizadas a pixeles
    points = []
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            x = int(coords[i] * img_width)
            y = int(coords[i + 1] * img_height)
            points.append([x, y])
    
    if len(points) < 3:
        return []
    
    # Crear contorno
    contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    return [contour]

def mouse_callback(event, x, y, flags, param):
    """Callback para manejar clicks del mouse"""
    if event == cv2.EVENT_LBUTTONDOWN:
        # Si se hace click en el boton "Siguiente"
        if param['button_rect'][0] <= x <= param['button_rect'][2] and \
           param['button_rect'][1] <= y <= param['button_rect'][3]:
            param['next_image'] = True

def procesar_imagen_aleatoria():
    """Procesa una imagen aleatoria y muestra la comparacion"""
    # Configurar rutas
    dataset_path = "/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/datasets/pebbles_final_filtered"
    train_images_path = os.path.join(dataset_path, "images/train")
    train_labels_path = os.path.join(dataset_path, "labels/train")
    
    # Obtener lista de imagenes
    image_files = glob.glob(os.path.join(train_images_path, "*.jpg"))
    
    if not image_files:
        print("No se encontraron imagenes en el dataset")
        return False
    
    # Seleccionar imagen aleatoria
    img_path = random.choice(image_files)
    img_name = Path(img_path).stem
    txt_path = os.path.join(train_labels_path, f"{img_name}.txt")
    
    print(f"Imagen seleccionada aleatoriamente: {img_name}")
    
    # Leer imagen
    img = cv2.imread(img_path)
    if img is None:
        print(f"No se pudo leer la imagen: {img_path}")
        return False
    
    img_height, img_width = img.shape[:2]
    
    # Leer archivo de texto
    if not os.path.exists(txt_path):
        print(f"No se encontro el archivo de texto: {txt_path}")
        return False
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Procesar cada linea de segmentacion
    all_contours = []
    for line in lines:
        contours = yolo_seg_to_contours(line, img_width, img_height)
        all_contours.extend(contours)
    
    if not all_contours:
        print(f"No se encontraron contornos validos en: {txt_path}")
        return False
    
    # Aplicar smoothing a todos los contornos
    smoothed_contours, epsilon_base, porcentaje = aplicar_smoothing(all_contours, img_width, img_height)
    
    # Crear imagen de comparacion
    img_original = img.copy()
    img_smoothed = img.copy()
    
    # Dibujar contornos originales en rojo (linea mas fina)
    cv2.drawContours(img_original, all_contours, -1, (0, 0, 255), 1)
    
    # Dibujar contornos suavizados en verde (linea mas fina)
    cv2.drawContours(img_smoothed, smoothed_contours, -1, (0, 255, 0), 1)
    
    # Crear imagen combinada (lado a lado) con mas espacio
    h, w = img.shape[:2]
    # Agregar espacio extra entre las imagenes
    combined_img = np.zeros((h, w * 2 + 50, 3), dtype=np.uint8)
    combined_img[:, :w] = img_original
    combined_img[:, w+50:] = img_smoothed
    
    # Agregar texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Texto para imagen original
    cv2.putText(combined_img, "ORIGINAL", (50, 50), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(combined_img, f"Contornos: {len(all_contours)}", (50, 100), font, 0.7, (255, 255, 255), 2)
    
    # Texto para imagen suavizada
    cv2.putText(combined_img, "SUAVIZADO", (w + 100, 50), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(combined_img, f"Contornos: {len(smoothed_contours)}", (w + 100, 100), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Epsilon: {epsilon_base:.4f}", (w + 100, 130), font, 0.6, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Ocupacion: {porcentaje:.1f}%", (w + 100, 160), font, 0.6, (255, 255, 255), 2)
    
    # Crear boton "Siguiente"
    button_y = h - 80
    button_height = 50
    button_width = 200
    button_x = (combined_img.shape[1] - button_width) // 2
    
    # Dibujar boton
    cv2.rectangle(combined_img, (button_x, button_y), (button_x + button_width, button_y + button_height), (100, 100, 100), -1)
    cv2.rectangle(combined_img, (button_x, button_y), (button_x + button_width, button_y + button_height), (255, 255, 255), 2)
    
    # Texto del boton
    button_text = "Siguiente Imagen"
    text_size = cv2.getTextSize(button_text, font, 0.8, 2)[0]
    text_x = button_x + (button_width - text_size[0]) // 2
    text_y = button_y + (button_height + text_size[1]) // 2
    cv2.putText(combined_img, button_text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
    
    # Mostrar estadisticas
    print(f"  - Contornos originales: {len(all_contours)}")
    print(f"  - Contornos suavizados: {len(smoothed_contours)}")
    print(f"  - Epsilon base: {epsilon_base:.4f}")
    print(f"  - Porcentaje de ocupacion del frame: {porcentaje:.1f}%")
    
    # Contar puntos en cada contorno
    total_points_original = sum(len(contour) for contour in all_contours)
    total_points_smoothed = sum(len(contour) for contour in smoothed_contours)
    print(f"  - Puntos totales originales: {total_points_original}")
    print(f"  - Puntos totales suavizados: {total_points_smoothed}")
    print(f"  - Reduccion de puntos: {((total_points_original - total_points_smoothed) / total_points_original * 100):.1f}%")
    
    # Configurar callback del mouse
    param = {
        'next_image': False,
        'button_rect': (button_x, button_y, button_x + button_width, button_y + button_height)
    }
    
    cv2.namedWindow("Comparacion: Original vs Suavizado")
    cv2.setMouseCallback("Comparacion: Original vs Suavizado", mouse_callback, param)
    
    # Mostrar imagen
    cv2.imshow("Comparacion: Original vs Suavizado", combined_img)
    
    # Esperar input
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Si se presiona ESC o se hace click en el boton
        if key == 27 or param['next_image']:
            break
        # Si se presiona 's' o 'n'
        elif key == ord('s') or key == ord('S'):
            param['next_image'] = True
            break
        elif key == ord('n') or key == ord('N'):
            break
    
    cv2.destroyAllWindows()
    return param['next_image']

def main():
    print("Script de comparacion de contornos con approxPolyDP adaptativo")
    print("=" * 60)
    print("Epsilon adaptativo por porcentaje de ocupacion del frame:")
    print("- 0% ocupacion: epsilon = 0.015")
    print("- 10%+ ocupacion: epsilon = 0.005")
    print("- Distribucion uniforme entre 0% y 10%")
    print("=" * 60)
    print("Instrucciones:")
    print("- Click en 'Siguiente Imagen' o presiona 's' para ver otra imagen")
    print("- Presiona 'n' o ESC para salir")
    print("=" * 60)
    
    while True:
        continuar = procesar_imagen_aleatoria()
        
        if not continuar:
            break
    
    print("¡Hasta luego!")

if __name__ == "__main__":
    main()
