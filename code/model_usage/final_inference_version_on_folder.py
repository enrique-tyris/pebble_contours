import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import csv
import argparse
from pathlib import Path
import sys

# Añadir el directorio padre al path para importar inference_yolo_adapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving_gradio.inference_yolo_adapter import process_yolo_on_image

def procesar_carpeta(input_folder, output_folder, confidence=0.5, area_factor=0.1, device='auto'):
    """
    Procesa todas las imágenes de una carpeta usando el mismo pipeline del servidor Gradio
    
    Args:
        input_folder (str): Ruta a la carpeta con las imágenes de entrada
        output_folder (str): Ruta a la carpeta donde guardar los resultados
        confidence (float): Umbral de confianza (default: 0.5)
        area_factor (float): Factor de área mínima (default: 0.1)
        device (str): Dispositivo a usar ('auto', 'cpu', 'gpu')
    """
    
    # Ruta al modelo entrenado (misma que en gradio_server.py)
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'pebble_seg_model.pt')
    model = YOLO(MODEL_PATH)
    model.to('cpu')
    
    # Crear carpetas de salida si no existen
    output_images_folder = os.path.join(output_folder, 'images')
    output_csv_folder = os.path.join(output_folder, 'csv')
    
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_csv_folder, exist_ok=True)
    
    # Extensiones de imagen soportadas
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Obtener lista de archivos de imagen
    input_path = Path(input_folder)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Encontradas {len(image_files)} imágenes para procesar")
    
    for i, image_file in enumerate(image_files, 1):
        print(f"Procesando {i}/{len(image_files)}: {image_file.name}")
        
        try:
            # Cargar imagen
            imagen_pil = Image.open(image_file)
            imagen_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
            
            # Procesar imagen usando la misma función del servidor Gradio
            resultado, coordenadas, poligonos = process_yolo_on_image(
                model,
                imagen_cv2,
                conf=confidence,
                device=device,
                area_factor=area_factor,
                return_coordinates=True,
                remove_overlapping=False
            )
            
            # Guardar imagen procesada
            output_image_path = os.path.join(output_images_folder, f"processed_{image_file.stem}.jpg")
            cv2.imwrite(output_image_path, resultado)
            
            # Guardar CSV con coordenadas
            output_csv_path = os.path.join(output_csv_folder, f"{image_file.stem}_coordinates.csv")
            
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Escribir encabezado
                writer.writerow(["Mask_ID", "Centroid_X", "Centroid_Y", "Polygon_Coordinates"])
                
                # Escribir coordenadas y polígonos
                for j, (coord, polygon) in enumerate(zip(coordenadas, poligonos)):
                    # Convertir polígono a string de coordenadas
                    polygon_str = ";".join([f"{point[0][0]},{point[0][1]}" for point in polygon])
                    writer.writerow([j+1, coord[0], coord[1], polygon_str])
            
            print(f"  ✓ Guardado: {output_image_path}")
            print(f"  ✓ CSV: {output_csv_path} ({len(coordenadas)} objetos detectados)")
            
        except Exception as e:
            print(f"  ✗ Error procesando {image_file.name}: {e}")
            continue
    
    print(f"\nProcesamiento completado. Resultados guardados en: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Procesar imágenes de una carpeta con el modelo YOLO')
    parser.add_argument('input_folder', help='Carpeta con las imágenes de entrada')
    parser.add_argument('output_folder', help='Carpeta donde guardar los resultados')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Umbral de confianza (default: 0.5)')
    parser.add_argument('--area_factor', type=float, default=0.1,
                       help='Factor de área mínima (default: 0.1)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                       help='Dispositivo a usar (default: auto)')
    
    args = parser.parse_args()
    
    # Verificar que la carpeta de entrada existe
    if not os.path.exists(args.input_folder):
        print(f"Error: La carpeta de entrada '{args.input_folder}' no existe")
        return
    
    # Procesar carpeta
    procesar_carpeta(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        confidence=args.confidence,
        area_factor=args.area_factor,
        device=args.device
    )

if __name__ == "__main__":
    main()
