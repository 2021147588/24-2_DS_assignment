import torch
import clip
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_from_disk
import torch.nn as nn

from model import CustomCLIPClassifier
from utils import CustomDataset, compute_ece, plot_confidence_and_accuracy, visualize_embeddings_with_tsne

# class label
class_label = [
        "antelope",
        "badger",
        "bat",
        "bear",
        "bee",
        "beetle",
        "bison",
        "boar",
        "butterfly",
        "cat",
        "caterpillar",
        "chimpanzee",
        "cockroach",
        "cow",
        "coyote",
        "crab",
        "crow",
        "deer",
        "dog",
        "dolphin",
        "donkey",
        "dragonfly",
        "duck",
        "eagle",
        "elephant",
        "flamingo",
        "fly",
        "fox",
        "goat",
        "goldfish",
        "goose",
        "gorilla",
        "grasshopper",
        "hamster",
        "hare",
        "hedgehog",
        "hippopotamus",
        "hornbill",
        "horse",
        "hummingbird",
        "hyena",
        "jellyfish",
        "kangaroo",
        "koala",
        "ladybugs",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "mosquito",
        "moth",
        "mouse",
        "octopus",
        "okapi",
        "orangutan",
        "otter",
        "owl",
        "ox",
        "oyster",
        "panda",
        "parrot",
        "pelecaniformes",
        "penguin",
        "pig",
        "pigeon",
        "porcupine",
        "possum",
        "raccoon",
        "rat",
        "reindeer",
        "rhinoceros",
        "sandpiper",
        "seahorse",
        "seal",
        "shark",
        "sheep",
        "snake",
        "sparrow",
        "squid",
        "squirrel",
        "starfish",
        "swan",
        "tiger",
        "turkey",
        "turtle",
        "whale",
        "wolf",
        "wombat",
        "woodpecker",
        "zebra"
      ]

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Load the saved model
model_load_path = "/root/Representational-Learning/saved_model.pth"
classifier_model = CustomCLIPClassifier(model).to(device)
classifier_model.load_state_dict(torch.load(model_load_path))
classifier_model.eval()
print(f"Model loaded from {model_load_path}")

# Load and preprocess data
dataset = load_from_disk("/root/Representational-Learning/dataset/val")
custom_dataset = CustomDataset(dataset, preprocess)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=False)

# Evaluate model and measure ECE
all_probs = []
all_labels = []
with torch.no_grad():
    for i, (images, labels) in enumerate(dataloader):
        if i == len(dataloader)-1:
            continue

        mapped_labels = [class_label[label]  for label in labels]
        
        images, labels = images.to(device), labels.to(device)
        
        text_tokens = clip.tokenize(mapped_labels).to("cuda")

        logits_per_image, logits_per_text = classifier_model(images, text_tokens)
        
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())


all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)




ece_score = compute_ece(all_probs, all_labels)
print(f"ECE Score: {ece_score:.4f}")
visualize_embeddings_with_tsne(classifier_model, dataloader)
plot_confidence_and_accuracy(all_probs, all_labels)

