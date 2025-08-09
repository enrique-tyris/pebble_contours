import cv2
import numpy as np
import os
from typing import Tuple, List

class AnnotationViewer:
    def __init__(self, directory: str):
        self.directory = directory
        self.current_index = 0
        self.image_files = self._get_image_files()
        
    def _get_image_files(self) -> List[str]:
        """Obtiene lista de imágenes que tienen archivo .txt asociado"""
        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = []
        
        for f in os.listdir(self.directory):
            if f.lower().endswith(valid_extensions):
                base_name = os.path.splitext(f)[0]
                txt_path = os.path.join(self.directory, base_name + '.txt')
                if os.path.exists(txt_path):
                    image_files.append(f)
        
        return sorted(image_files)

    def visualizar_anotaciones_yolo(self, imagen_path: str, anotacion_path: str) -> np.ndarray:
        """Visualiza las anotaciones YOLO sobre una imagen"""
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {imagen_path}")
        
        altura, ancho = imagen.shape[:2]
        
        try:
            with open(anotacion_path, 'r') as f:
                anotaciones = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"No se pudo encontrar el archivo de anotaciones: {anotacion_path}")
        
        imagen_anotada = imagen.copy()
        
        for anotacion in anotaciones:
            valores = anotacion.strip().split()
            clase_id = int(valores[0])
            
            puntos = np.array([float(x) for x in valores[1:]])
            puntos = puntos.reshape(-1, 2)
            
            puntos[:, 0] *= ancho
            puntos[:, 1] *= altura
            
            puntos = puntos.astype(np.int32)
            
            color = (0, 255, 0)  # Verde
            cv2.polylines(imagen_anotada, [puntos], True, color, 2)
            
            overlay = imagen_anotada.copy()
            cv2.fillPoly(overlay, [puntos], color)
            cv2.addWeighted(overlay, 0.3, imagen_anotada, 0.7, 0, imagen_anotada)
        
        return imagen_anotada

    def mostrar_imagen_actual(self) -> None:
        """Muestra la imagen actual con sus anotaciones"""
        if not self.image_files:
            print("No se encontraron imágenes con anotaciones")
            return

        imagen_file = self.image_files[self.current_index]
        base_name = os.path.splitext(imagen_file)[0]
        
        imagen_path = os.path.join(self.directory, imagen_file)
        anotacion_path = os.path.join(self.directory, base_name + '.txt')
        
        try:
            imagen_anotada = self.visualizar_anotaciones_yolo(imagen_path, anotacion_path)
            
            # Mostrar información en la imagen
            total_imagenes = len(self.image_files)
            info_text = f"Imagen {self.current_index + 1}/{total_imagenes}: {imagen_file}"
            cv2.putText(imagen_anotada, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Visor de Anotaciones YOLO', imagen_anotada)
            
        except Exception as e:
            print(f"Error al procesar {imagen_file}: {str(e)}")

    def navegar(self) -> None:
        """Inicia la navegación interactiva entre imágenes"""
        if not self.image_files:
            print("No se encontraron imágenes con anotaciones")
            return
        
        print("\nControles:")
        print("→ o 'd': Siguiente imagen")
        print("← o 'a': Imagen anterior")
        print("'q': Salir")
        
        self.mostrar_imagen_actual()
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key in [ord('d'), 83]:  # Derecha
                self.current_index = (self.current_index + 1) % len(self.image_files)
                self.mostrar_imagen_actual()
            elif key in [ord('a'), 81]:  # Izquierda
                self.current_index = (self.current_index - 1) % len(self.image_files)
                self.mostrar_imagen_actual()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Directorio con las imágenes y anotaciones
    DIRECTORY = "/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/seleccion"
    
    viewer = AnnotationViewer(DIRECTORY)
    viewer.navegar()
