import torch
from models import MnistAutoencoder, CIFAR10Autoencoder, FashionMnistAutoencoder, RetinaAutoencoder
import pandas as pd
import random
import os
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch



def load_mnist_autoencoder(model_path):
    model = MnistAutoencoder()
    model.load_state_dict(torch.load(model_path))
    return model

def load_fashion_mnist_autoencoder(model_path):
    model = FashionMnistAutoencoder()
    model.load_state_dict(torch.load(model_path))
    return model

def load_cifar10_autoencoder(model_path):
    model = CIFAR10Autoencoder()
    model.load_state_dict(torch.load(model_path))
    return model

def load_retina_autoencoder(model_path):
    model = RetinaAutoencoder()
    model.load_state_dict(torch.load(model_path))
    return model

def get_labels(csv_to_read):
    random.seed(42)
    df = pd.read_csv(csv_to_read)
    column_names = df.columns.tolist()
    column_names.append("NDP")
    column_names = column_names[2:]
    rows = []
    labels = []
    for row in df.values:
        class_vector = row[2:].tolist()
        if row[1] == 1:
            class_vector.append(0)
        elif row[1] == 0 and sum(class_vector) == 0:
            class_vector.append(1)
        elif row[1] == 0 and sum(class_vector) != 0:
            class_vector.append(0)

        indices = [index for index, value in enumerate(class_vector) if value == 1]
        selected_index = random.choice(indices)
        labels.append(selected_index)
    return column_names, labels

def read_images(image_directory):
    images_tensor = []
    images = os.listdir(image_directory)
    images.sort(key=lambda x: int(x.split('.')[0]))
    for image in images:
        try:
            image_name = image_directory+image
            img = read_image(image_name)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ConvertImageDtype(torch.float),
            ])
            img_tensor = transform(img)
            images_tensor.append(img_tensor)
        except FileNotFoundError:
            print(f"File not found: {image}")
    return images_tensor
