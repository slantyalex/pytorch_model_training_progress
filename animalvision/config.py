import torch
import os

class Config:
    DATASET_NAME = "CIFAR10" 
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 128
    NUM_WORKERS = 2

    NUM_CLASSES = 10
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    MODEL_SAVE_PATH = os.path.join(DATA_DIR, "best_model.pth")