from datetime import datetime

class ModelConfig:
    EPOCHS = 150
    BATCH_SIZE = 16
    IMG_SIZE = 224
    IMG_HEIGHT = 128
    IMG_WIDTH = 256
    IMG_DIR = "dataset/data"
    MODEL_DIR = "saved_models"
    LOG_DIR = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")


