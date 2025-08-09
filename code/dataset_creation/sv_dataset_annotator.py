import os
import torch
import gc
from sv_etiquetador_sam import detect_all_with_supervision
from tqdm import tqdm
from datetime import datetime

def process_directory(
    input_dir: str,
    checkpoint_path: str,
    iou_threshold: float = 0.25,
    valid_extensions: tuple = ('.jpg', '.jpeg', '.png')
):
    """
    Procesa todas las imágenes en un directorio usando SAM para generar anotaciones YOLO.
    
    Args:
        input_dir: Directorio con las imágenes a procesar
        checkpoint_path: Ruta al archivo de pesos de SAM
        iou_threshold: Umbral IoU para NMS
        valid_extensions: Extensiones de archivo válidas
    """
    # Crear archivo de log para errores
    log_file = os.path.join(input_dir, f"errores_procesamiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Verificar que el directorio existe
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"El directorio '{input_dir}' no existe")

    # Crear sets de nombres sin extensión
    image_names = {os.path.splitext(f)[0] for f in os.listdir(input_dir) 
                  if f.lower().endswith(valid_extensions)}
    txt_names = {os.path.splitext(f)[0] for f in os.listdir(input_dir) 
                if f.lower().endswith('.txt')}
    
    # Encontrar imágenes que no tienen txt
    pending_images = image_names - txt_names
    
    # Convertir de nuevo a nombres de archivo con extensión
    pending_files = []
    for name in pending_images:
        for ext in valid_extensions:
            full_name = name + ext
            if os.path.exists(os.path.join(input_dir, full_name)):
                pending_files.append(full_name)
                break

    print(f"Encontradas {len(image_names)} imágenes en total")
    print(f"Ya procesadas: {len(txt_names)} imágenes")
    print(f"Pendientes de procesar: {len(pending_files)} imágenes")

    # Procesar cada imagen pendiente con barra de progreso
    for image_file in tqdm(pending_files, desc="Procesando imágenes"):
        image_path = os.path.join(input_dir, image_file)
        try:
            # Llamar a la función de detección sin visualizaciones
            detect_all_with_supervision(
                image_path=image_path,
                checkpoint=checkpoint_path,
                iou_threshold=iou_threshold,
                visualize=False  # Desactivar visualizaciones
            )
            
            # Limpieza agresiva de memoria
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()  # Forzar recolección de basura
            
        except Exception as e:
            error_msg = f"Error procesando {image_file}: {str(e)}"
            print(error_msg)
            # Registrar el error en el archivo de log
            with open(log_file, 'a') as f:
                f.write(f"{error_msg}\n")
            
            # Intentar limpiar memoria incluso después de un error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    INPUT_DIR = "/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/seleccion"
    SAM_CHECKPOINT = "/home/enrique/Desktop/VARIOS/garzIA/SURF ANALYST/code/wave_segmentation/weights/sam_vit_h_4b8939.pth"
    
    process_directory(
        input_dir=INPUT_DIR,
        checkpoint_path=SAM_CHECKPOINT,
        iou_threshold=0.25
    )
