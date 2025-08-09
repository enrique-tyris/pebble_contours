import cv2
import numpy as np

def visualizar_anotaciones_yolo(imagen_path, anotacion_path):
    # Leer la imagen
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {imagen_path}")
    
    altura, ancho = imagen.shape[:2]
    
    # Leer el archivo de anotaciones
    try:
        with open(anotacion_path, 'r') as f:
            anotaciones = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"No se pudo encontrar el archivo de anotaciones: {anotacion_path}")
    
    # Crear una copia de la imagen para dibujar
    imagen_anotada = imagen.copy()
    
    # Procesar cada línea de anotación
    for anotacion in anotaciones:
        # Separar los valores
        valores = anotacion.strip().split()
        clase_id = int(valores[0])
        
        # Convertir las coordenadas normalizadas a píxeles
        puntos = np.array([float(x) for x in valores[1:]])
        puntos = puntos.reshape(-1, 2)
        
        # Desnormalizar coordenadas
        puntos[:, 0] *= ancho
        puntos[:, 1] *= altura
        
        # Convertir a enteros para dibujar
        puntos = puntos.astype(np.int32)
        
        # Dibujar el polígono
        color = (0, 255, 0)  # Verde
        cv2.polylines(imagen_anotada, [puntos], True, color, 2)
        
        # Rellenar el polígono con transparencia
        overlay = imagen_anotada.copy()
        cv2.fillPoly(overlay, [puntos], color)
        cv2.addWeighted(overlay, 0.3, imagen_anotada, 0.7, 0, imagen_anotada)
    
    # Mostrar la imagen
    cv2.imshow('Anotaciones YOLO', imagen_anotada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return imagen_anotada

# Ejemplo de uso
if __name__ == "__main__":
    # Reemplaza estas rutas con tus archivos
    imagen_path = "/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/Pebble2.jpeg"
    anotacion_path = "/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/Pebble2.txt"
    
    imagen_resultado = visualizar_anotaciones_yolo(imagen_path, anotacion_path)
