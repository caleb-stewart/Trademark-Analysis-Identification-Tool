def resource_path(relative_path):
    import os
    import sys

    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
        full_path = os.path.join(base_path, "_internal", "models", relative_path)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_path, relative_path)
    return full_path

_model = None

def initialize_model():
    import os
    from ultralytics import YOLO

    global _model
    if _model is None:
        model_path = resource_path("logo_detection.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _model = YOLO(model_path)
    return _model