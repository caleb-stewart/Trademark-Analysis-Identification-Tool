import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as tr
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BeitImageProcessor, BeitModel

class EmbeddingExtractor:
    """Class for extracting image embeddings using ResNet-50, CLIP, and BEiT."""

    def __init__(self, device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load ResNet-50 model
        self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.resnet = self.resnet.to(self.device).eval()

        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Load BEiT model
        self.beit_model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224").to(self.device).eval()
        self.beit_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")

    def preprocess_resnet(self, img):
        transformations = tr.Compose([
            tr.Resize((224, 224)),
            tr.ToTensor(),
            tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = transformations(img).unsqueeze(0).to(self.device)
        return img

    def preprocess_clip(self, img):
        return self.clip_processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)

    def preprocess_beit(self, img):
        return self.beit_processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)

    def get_embedding(self, img, model_name="resnet"):

        img = Image.fromarray(img)

        if model_name == "resnet":
            img_tensor = self.preprocess_resnet(img)
            with torch.no_grad():
                embedding = self.resnet(img_tensor).cpu().numpy()

        elif model_name == "clip":
            img_tensor = self.preprocess_clip(img)
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(img_tensor).cpu().numpy()

        elif model_name == "beit":
            img_tensor = self.preprocess_beit(img)
            with torch.no_grad():
                embedding = self.beit_model(img_tensor).last_hidden_state.mean(dim=1).cpu().numpy()

        else:
            raise ValueError("Invalid model name. Choose from: resnet, clip, beit.")

        return embedding

    @staticmethod
    def cosine_similarity(emb1, emb2):

        return torch.nn.functional.cosine_similarity(torch.tensor(emb1), torch.tensor(emb2)).item()

    @staticmethod
    def euclidean_distance(emb1, emb2):

        return np.linalg.norm(emb1 - emb2)