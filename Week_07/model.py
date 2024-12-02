import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(512, 90)  # Assuming 90 classes, adjust accordingly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def forward(self, images, text_tokens):

        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        
        
        return logits_per_image, logits_per_text
   