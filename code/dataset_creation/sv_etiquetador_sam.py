import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import supervision as sv

def check_multiple_segments(mask_binary):
    """Comprueba si una máscara tiene múltiples segmentos usando findContours"""
    contours, _ = cv2.findContours(
        mask_binary.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    return len(contours) > 1

def get_largest_segment(mask_binary):
    """Retorna la máscara del segmento de mayor área"""
    # Encontrar todos los segmentos conectados
    num_labels, labels = cv2.connectedComponents(mask_binary.astype(np.uint8))
    
    if num_labels == 1:  # Solo el fondo
        return None
    
    # Encontrar el segmento más grande (excluyendo el fondo, que es 0)
    largest_label = 1
    largest_area = 0
    
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area > largest_area:
            largest_area = area
            largest_label = label
    
    # Crear máscara con solo el segmento más grande
    return labels == largest_label

def process_mask(mask):
    """Procesa una máscara según los criterios establecidos"""
    mask_binary = mask['segmentation'].astype(np.uint8)
    
    # Comprobar múltiples segmentos
    if check_multiple_segments(mask_binary):
        # Quedarse solo con el segmento más grande
        largest_segment = get_largest_segment(mask_binary)
        if largest_segment is None:
            return None
        
        mask['segmentation'] = largest_segment
    
    return mask

def detect_all_with_supervision(
    image_path: str,
    model_type: str = "vit_h",
    checkpoint: str = "path/to/sam_vit_h.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    iou_threshold: float = 0.5,
    visualize: bool = True
):
    # 1) Carga del modelo SAM
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device, non_blocking=True)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 2) Leer imagen
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen '{image_path}'")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3) Generar todas las máscaras
    sam_result = mask_generator.generate(image_rgb)
    print(f"Generadas {len(sam_result)} máscaras iniciales.")

    # 4) Preparar datos para NMS
    # Convertir máscaras a formato necesario para NMS
    masks = np.array([mask['segmentation'] for mask in sam_result])
    
    # Crear array de predictions usando los bounding boxes y scores
    predictions = np.array([
        [
            mask['bbox'][0],  # x_min
            mask['bbox'][1],  # y_min
            mask['bbox'][0] + mask['bbox'][2],  # x_max
            mask['bbox'][1] + mask['bbox'][3],  # y_max
            mask['predicted_iou']  # score
        ] for mask in sam_result
    ])

    # Aplicar NMS
    keep_mask = sv.mask_non_max_suppression(
        predictions=predictions,
        masks=masks,
        iou_threshold=iou_threshold
    )

    # Filtrar resultados
    sam_result_filtered = [mask for idx, mask in enumerate(sam_result) if keep_mask[idx]]
    print(f"Quedan {len(sam_result_filtered)} máscaras después de NMS.")

    # Procesar cada máscara
    sam_result_processed = []
    for mask in sam_result_filtered:
        processed_mask = process_mask(mask)
        if processed_mask is not None:
            sam_result_processed.append(processed_mask)
    
    print(f"Quedan {len(sam_result_processed)} máscaras después del procesamiento")

    # Convertir a detecciones de supervision
    detections = sv.Detections.from_sam(sam_result=sam_result_processed)

    # Guardar en formato YOLO
    image_height, image_width = image_bgr.shape[:2]
    output_path = os.path.splitext(image_path)[0] + '.txt'
    
    with open(output_path, 'w') as f:
        for mask in sam_result_processed:
            # Convertir máscara a polígonos
            polygons = sv.mask_to_polygons(mask['segmentation'])
            
            # Para cada polígono encontrado
            for polygon in polygons:
                # Normalizar coordenadas
                normalized_polygon = polygon.astype(float)
                normalized_polygon[:, 0] /= image_width
                normalized_polygon[:, 1] /= image_height
                
                # Convertir a formato YOLO
                points_str = ' '.join([f"{x} {y}" for x, y in normalized_polygon])
                f.write(f"0 {points_str}\n")

    print(f"Archivo de anotación YOLO guardado en: {output_path}")

    if visualize:
        # Visualización final
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_image = mask_annotator.annotate(
            scene=image_bgr.copy(),
            detections=detections
        )

        # Mostrar resultado
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Imagen original
        axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Segmentación final
        axes[1].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Segmentación Final ({len(sam_result_processed)} máscaras)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    return sam_result_processed

if __name__ == "__main__":
    IMG_PATH = "/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/Pebble2.jpeg"
    SAM_CHECKPOINT = "/home/enrique/Desktop/VARIOS/garzIA/SURF ANALYST/code/wave_segmentation/weights/sam_vit_h_4b8939.pth"
    detect_all_with_supervision(IMG_PATH, checkpoint=SAM_CHECKPOINT, iou_threshold=0.25)
