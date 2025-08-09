import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funcion principal de procesamiento
def process_image(image, blur_param, canny_params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Solo aplicar Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (blur_param, blur_param), 0)

    # Aplicar Canny
    edges = cv2.Canny(blurred, canny_params[0], canny_params[1])

    # Encontrar y dibujar contornos
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# ---- Configuraciones ----

# Lista de nombres de archivos
image_files = ["data/Pebble1.jpeg", "data/Pebble2.jpeg", "data/Pebble3.jpeg"]

# Parámetros
blur_params_refined = [5, 7, 9]
canny_configs_around_best = [(110, 210), (120, 220), (130, 230)]

# Procesar cada imagen
for image_path in image_files:
    image = cv2.imread(image_path)

    # Crear figura para cada imagen
    fig, axs = plt.subplots(len(blur_params_refined), len(canny_configs_around_best), figsize=(18, 12))
    fig.suptitle('Processing {}'.format(image_path), fontsize=18)

    for i, blur_param in enumerate(blur_params_refined):
        for j, canny_param in enumerate(canny_configs_around_best):
            processed_image = process_image(image, blur_param, canny_param)
            axs[i, j].imshow(processed_image)
            axs[i, j].set_title('Gaussian {}, Canny {}'.format(blur_param, canny_param))
            axs[i, j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()  # Mostrar una ventana por imagen
