# gradio_server.py
import gradio as gr
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from inference_yolo_adapter import process_yolo_on_image
import tempfile
import os

# Ruta a tu modelo entrenado
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'pebble_seg_model.pt')
model = YOLO(MODEL_PATH)
model.to('cpu')

# Variable global para almacenar las coordenadas y polígonos
coordenadas_actuales = []
poligonos_actuales = []

# Función Gradio
def inferir(imagen_pil, confidence=0.5, area_factor=0.1, device='auto'):
    global coordenadas_actuales, poligonos_actuales
    
    imagen_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    resultado, coordenadas, poligonos = process_yolo_on_image(model,
                                                             imagen_cv2,
                                                             conf=confidence,
                                                             device=device,
                                                             area_factor=area_factor,
                                                             return_coordinates=True,
                                                             remove_overlapping=False)
    
    coordenadas_actuales = coordenadas
    poligonos_actuales = poligonos
    
    return Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))

def descargar_coordenadas():
    """Genera y devuelve el archivo de coordenadas en formato CSV con polígonos"""
    global coordenadas_actuales, poligonos_actuales
    
    if not coordenadas_actuales:
        return None
    
    # Crear archivo temporal
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    
    try:
        # Escribir encabezado CSV
        temp_file.write("Mask_ID,Centroid_X,Centroid_Y,Polygon_Coordinates\n")
        
        # Escribir coordenadas y polígonos
        for i, (coord, polygon) in enumerate(zip(coordenadas_actuales, poligonos_actuales)):
            # Convertir polígono a string de coordenadas
            polygon_str = ";".join([f"{point[0][0]},{point[0][1]}" for point in polygon])
            temp_file.write(f"{i+1},{coord[0]},{coord[1]},\"{polygon_str}\"\n")
        
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"Error al crear archivo: {e}")
        return None

# Interface
with gr.Blocks(title="Pebble segmentation") as demo:
    gr.Markdown("# Pebble Segmentation")
    gr.Markdown("This server uses a YOLOv8 model for object segmentation with adaptive morphological post-processing.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload or drag an image")
            confidence_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Confidence Threshold")
            area_factor_slider = gr.Slider(minimum=0.05, maximum=0.5, value=0.1, step=0.05, label="Minimum Area Factor")
            device_dropdown = gr.Dropdown(choices=['auto', 'cpu', 'gpu'], value='auto', label="Device")
            infer_btn = gr.Button("Process Image", variant="primary")
        
        with gr.Column():
            image_output = gr.Image(type="pil", label="Result with segmentation")
            download_btn = gr.Button("Download Coordinates", variant="secondary")
            file_output = gr.File(label="Coordinates File (CSV)", visible=False, interactive=False)
    
    # Eventos
    infer_btn.click(
        fn=inferir,
        inputs=[image_input, confidence_slider, area_factor_slider, device_dropdown],
        outputs=image_output
    )
    
    download_btn.click(
        fn=descargar_coordenadas,
        outputs=file_output
    ).then(
        lambda: gr.File(visible=True, interactive=True), # Hacemos visible e interactivo el campo de archivo
        outputs=file_output
    )

if __name__ == "__main__":
    demo.launch(share=True)
