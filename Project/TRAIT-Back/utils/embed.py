import os

current_dir = os.getcwd()
cache_dir = os.path.join(current_dir, "model_cache")
os.makedirs(cache_dir, exist_ok=True)

class BEiTEmbedding:
    def __init__(self, model_name="microsoft/beit-base-patch16-224"):
        import logging

        from transformers import AutoImageProcessor, BeitModel

        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir) 
        self.model = BeitModel.from_pretrained(model_name, cache_dir=cache_dir)

    def extract_embedding(self, img):
        import torch

        inputs = self.feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


class CLIPEmbedding:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    def extract_embedding(self, img):
        import torch

        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.squeeze().numpy()


class ResNetEmbedding:
    def __init__(self):
        import torch.nn as nn
        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_embedding(self, img):
        import torch

        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img)
        return embedding.squeeze().cpu().numpy()