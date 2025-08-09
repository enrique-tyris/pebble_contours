from ultralytics import YOLO

# Load the YOLOv8 model for segmentation
model = YOLO('yolov8s-seg.pt')  # You can also use yolov8s-seg.pt, yolov8m-seg.pt, etc.

# Set up training parameters
training_parameters = {
    'data': '/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/datasets/pebbles_v2_super_filtered/data_config.yaml',  # Path to your dataset YAML file
    'epochs': 150,                                      # Number of training epochs
    'imgsz': 640,                                      # Image size
    'batch': 6,                                       # Batch size, adjust according to your GPU memory
    'augment': True,                                   # Enable default augmentations
    'mosaic': 0,                                       # Enable mosaic augmentation
    'close_mosaic': 0,                                # Disable mosaic before 85 epochs
    'degrees': 20,                                     # Rotation augmentation (±10 degrees)
    'translate': 0.1,                                  # Translation augmentation (10% translation)
    'scale': 0.5,                                      # Scale augmentation (scales images by ±50%)
    'shear': 0.05,                                      # Shear augmentation (±2 degrees)
    'flipud': 0.5,                                     # Vertical flip augmentation with 50% probability
    'fliplr': 0.5,                                     # Horizontal flip augmentation with 50% probability
    'hsv_h': 0.10,                                    # Hue shift augmentation (±1.5%)
    'hsv_s': 0.6,                                      # Saturation augmentation (±70%)
    'hsv_v': 0.5,                                      # Value (brightness) augmentation (±40%)
    'patience': 25,                                    # Early stopping patience (stop if no improvement for 10 epochs)
    'name': 'pebbles_v2_super_filtered_MEDIUM_AUG_NO_MOSAICeven_more_epochs'                          # Custom name for this training run
}

# Train the model
model.train(**training_parameters)



# Como estaban antes algunos params
#################################################################################################
#    'degrees': 30,                                     # Rotation augmentation (±10 degrees)
#    'translate': 0.2,                                  # Translation augmentation (10% translation)
#    'shear': 2.0,                                      # Shear augmentation (±2 degrees)
#    'hsv_h': 0.085,                                    # Hue shift augmentation (±1.5%)
#    'hsv_s': 0.8,                                      # Saturation augmentation (±70%)
