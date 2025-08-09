# Pebble Contours

Pebble contour detection and segmentation system using YOLOv8.

## Project Structure

```
pebble_contours/
├── code/                      # Source code
│   ├── dataset_creation/      # Dataset creation and processing scripts
│   ├── model_training/        # Training scripts
│   ├── model_usage/          # Inference scripts
│   └── serving_gradio/       # Gradio server for web interface
├── data/                      # Data and resources
│   ├── datasets/             # Training datasets
│   ├── initial/             # Initial test images
│   ├── results/             # Inference results
│   └── test/                # Test images
└── runs/                     # Training results
    └── segment/             # Trained models
```

## Requirements

- Python 3.8+
- Anaconda
- YOLOv8
- Gradio
- Other dependencies (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [REPOSITORY_URL]
cd pebble_contours
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a new model:

```bash
python code/model_training/yolotrain.py
```

### Inference

To perform inference on a single image:

```bash
python code/model_usage/inference_v2_single_image.py
```

To process multiple images at once:

```bash
python code/model_usage/inference_v2_multiple_images.py
```

### Web Interface

To start the Gradio server:

```bash
python code/serving_gradio/gradio_server.py
```

## Author
Enrique García Iglesias