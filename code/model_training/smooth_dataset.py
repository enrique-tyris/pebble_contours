import os
import glob
import shutil
import cv2
import numpy as np
from pathlib import Path

# --- Funciones de suavizado adaptativo ---

def calcular_epsilon_por_porcentaje_frame(contours, img_w, img_h):
    min_eps, max_eps = 0.005, 0.012
    frame_area = img_w * img_h
    total_area = sum(cv2.contourArea(c) for c in contours)
    pct = (total_area / frame_area) * 100
    normalized = 1.0 if pct >= 5.0 else pct / 5.0
    eps_base = max_eps - (max_eps - min_eps) * normalized
    return eps_base

def aplicar_smoothing(contours, img_w, img_h):
    eps_base = calcular_epsilon_por_porcentaje_frame(contours, img_w, img_h)
    smoothed = []
    for c in contours:
        eps = eps_base * cv2.arcLength(c, True)
        smoothed.append(cv2.approxPolyDP(c, eps, True))
    return smoothed

# --- Conversión YOLOseg <-> contorno ---

def yolo_seg_to_contours(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 9: 
        return [], None
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    pts = []
    for i in range(0, len(coords), 2):
        x = int(coords[i]   * img_w)
        y = int(coords[i+1] * img_h)
        pts.append([x, y])
    if len(pts) < 3:
        return [], cls
    contour = np.array(pts, dtype=np.int32).reshape(-1,1,2)
    return [contour], cls

def contour_to_yolo_seg(contour, cls, img_w, img_h):
    pts = contour.reshape(-1,2)
    # Normalizar y aplanar
    norm = []
    for x,y in pts:
        norm += [x / img_w, y / img_h]
    return " ".join([cls] + [f"{v:.6f}" for v in norm])

# --- Procesamiento global del dataset ---

def procesar_dataset(origen_root, destino_root):
    # Define subcarpetas
    for split in ["train", "val"]:
        imgs_in  = os.path.join(origen_root, "images", split)
        lbls_in  = os.path.join(origen_root, "labels", split)
        imgs_out = os.path.join(destino_root, "images", split)
        lbls_out = os.path.join(destino_root, "labels", split)
        os.makedirs(imgs_out, exist_ok=True)
        os.makedirs(lbls_out, exist_ok=True)

        # Copiar imágenes
        for img_path in glob.glob(os.path.join(imgs_in, "*.jpg")):
            shutil.copy2(img_path, imgs_out)

        # Procesar etiquetas
        for lbl_path in glob.glob(os.path.join(lbls_in, "*.txt")):
            img_name = Path(lbl_path).stem + ".jpg"
            img = cv2.imread(os.path.join(imgs_in, img_name))
            if img is None:
                continue
            h,w = img.shape[:2]
            nuevas_lineas = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    contours, cls = yolo_seg_to_contours(line, w, h)
                    if not contours:
                        continue
                    smooth = aplicar_smoothing(contours, w, h)
                    # puede haber múltiples contornos; los escribimos uno a uno
                    for c in smooth:
                        nuevas_lineas.append(contour_to_yolo_seg(c, cls, w, h))

            # Guardar nuevo .txt
            with open(os.path.join(lbls_out, Path(lbl_path).name), 'w') as f_out:
                f_out.write("\n".join(nuevas_lineas))

if __name__ == "__main__":
    origen = "/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/datasets/pebbles_final_filtered"
    destino = "/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/datasets/pebbles_smoothed"
    print(f"Creando copia suavizada en:\n  {destino}")
    procesar_dataset(origen, destino)
    print("¡Listo! Dataset duplicado con contornos suavizados.")
