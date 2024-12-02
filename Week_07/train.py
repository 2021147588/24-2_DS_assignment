import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from model import CustomCLIPClassifier
from utils import CustomDataset

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

# Initialize custom classifier model
classifier_model = CustomCLIPClassifier(model).to(device)
optimizer = torch.optim.Adam(classifier_model.classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

dataset = load_from_disk("/root/Representational-Learning/dataset/train")
custom_dataset = CustomDataset(dataset, preprocess)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# Training loop
classifier_model.train()

for epoch in range(5):  # Train for 5 epochs, adjust as needed
    epoch_loss = 0
    for images, labels in tqdm(dataloader):
        mapped_labels = [class_label[label] for label in labels]

        images, labels = images.to(device), labels.to(device)
        text_tokens = clip.tokenize(mapped_labels).to("cuda")

        logits_per_image, logits_per_text = classifier_model(images, text_tokens)
        
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

# Save the trained model
model_save_path = "/root/Representational-Learning/saved_model.pth"
torch.save(classifier_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
