from multiprocessing import freeze_support
import os
import signal
import threading

from app import create_app

app = create_app()

def warmup():
    print("Loading major imports...")
    import cv2
    from flask import jsonify
    import numpy
    from PIL import Image
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
    import transformers
    from transformers import AutoImageProcessor, BeitModel, CLIPModel, CLIPProcessor
    from ultralytics import YOLO
    from models.model_load import initialize_model
    initialize_model()
    print("Major loading complete.")

def handle_shutdown(signum, frame):
    import sys

    print("Shutting down Flask backend gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if __name__ == "__main__":
    # Change the environment to "dev" for development, "prod" for production  
    env = os.environ.get("ENV", "prod")
    port = int(os.environ.get("PORT", 5174))

    freeze_support()
    threading.Thread(target=warmup, daemon=True).start()

    if env == "prod":
        from waitress import serve

        print(f"[PROD] Starting backend on http://127.0.0.1:{port} with waitress...")
        serve(app, host="127.0.0.1", port=port)
    else:
        print(f"[DEV] Starting backend on http://127.0.0.1:{port} with Flask...")
        app.run(host="127.0.0.1", port=port, debug=True, use_reloader=True)