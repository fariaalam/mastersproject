import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist_autoencoder,load_cifar10_autoencoder, load_fashion_mnist_autoencoder, load_retina_autoencoder


def train(train_dataloader, val_dataloader, epochs, model, model_path, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_loss = 1000
    for epoch in range(epochs):
        model.train()
        for index, data in enumerate(train_dataloader):
            images, _ = data
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print(f"Training -- Epoch : {epoch + 1} Iteration : {index+1} Loss : {loss.item():.5f}")

        model.eval()
        val_loss = 0
        for index, data in enumerate(val_dataloader):
            images, _ = data
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss = val_loss + loss.item()
            if index % 10 == 0:
                print(f"Validation -- Epoch : {epoch + 1} Iteration : {index+1} Loss : {loss.item():.5f}")
        
        if val_loss < best_loss:
            torch.save(model.state_dict(), model_path)
            best_loss = val_loss


def get_features(model, train_data, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    labled_features = []
    true_labels = []
    for data in train_data:
        images, labels = data
        images = images.to(device)
        true_labels.extend(labels)
        outputs = model.encoder(images)
        if device=="cpu":
            encoded_features = outputs.detach().numpy().tolist()
        else:
            encoded_features = outputs.cpu().detach().numpy().tolist()
        labled_features.extend(encoded_features)

    unlabled_features = []
    original_labels = []
    for data in test_data:
        images, labels = data
        images = images.to(device)
        original_labels.extend(labels)
        outputs = model.encoder(images)
        if device=="cpu":
            encoded_features = outputs.detach().numpy().tolist()
        else:
            encoded_features = outputs.cpu().detach().numpy().tolist()
        unlabled_features.extend(encoded_features)

    return labled_features, true_labels, unlabled_features, original_labels

def get_label_names(original_labels, predictions, class_names):
    original_label_names =[]
    prediction_label_names = []
    for label in original_labels:
        original_label_names.append(class_names[label])
    for label in predictions:
        prediction_label_names.append(class_names[label])
    return original_label_names, prediction_label_names

def train_test_svc(features, lables, unlabeled_features, original_labels, class_names, image_path):
    clf = SVC()
    clf.fit(features, lables)
    predictions = clf.predict(unlabeled_features)
    f1score = f1_score(original_labels, predictions, average='micro')
    accuracy = accuracy_score(original_labels, predictions)
    report = classification_report(original_labels, predictions, target_names=class_names)
    original_names, predicted_names = get_label_names(original_labels, predictions, class_names)
    conf_matrix = confusion_matrix(original_names, predicted_names, labels=class_names)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    display.plot()
    plt.xticks(rotation=90)
    plt.savefig(image_path, bbox_inches="tight")
    return f1score, accuracy, report 

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def find_closest_vector_indexes(feature, vectors, top_n=5):
    distances = []
    for i, vector in enumerate(vectors):
        distance = 1 - cosine_similarity(feature, vector)
        distances.append((distance, i))
    distances.sort(key=lambda x: x[0])
    return [i for _, i in distances[:top_n]]

def precision_at_k(retrieved_labels, original_label, k):
    relevant_count = sum(1 for label in retrieved_labels[:k] if label == original_label)
    return relevant_count / k

def average_precision(retrieved_labels, original_label):
    relevant_positions = [i + 1 for i, label in enumerate(retrieved_labels) if label == original_label]
    if not relevant_positions:
        return 0.0
    
    precisions = [precision_at_k(retrieved_labels, original_label, k) for k in relevant_positions]
    return sum(precisions) / len(relevant_positions)

def mean_average_precision(all_retrieved_labels, all_original_labels):
    average_precisions = [average_precision(retrieved, original) 
                          for retrieved, original in zip(all_retrieved_labels, all_original_labels)]
    return sum(average_precisions) / len(average_precisions)

def mean_reciprocal_rank(all_retrieved_labels, all_original_labels):
    reciprocal_ranks = []
    for retrieved, original in zip(all_retrieved_labels, all_original_labels):
        for rank, label in enumerate(retrieved, start=1):
            if label == original:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def train_evaluate(epochs, batch_size, test_size, seed, model_path, dataset_name,
                         train_dataloader, val_dataloader, semi_train_dataloader, semi_test_dataloader,
                         model, learning_rate, class_names, image_path):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(train_dataloader, val_dataloader, epochs, model, model_path, criterion, optimizer)

    if dataset_name == "mnist":
        model = load_mnist_autoencoder(model_path)
    elif dataset_name == "fashion_mnist":
        model = load_fashion_mnist_autoencoder(model_path)
    elif dataset_name == "cifat10":
        model = load_cifar10_autoencoder(model_path)
    elif dataset_name == "retina":
        model = load_retina_autoencoder(model_path)
    
    labled_features, true_labels, unlabeled_features, original_labels = get_features(model, 
                                                                                    semi_train_dataloader, 
                                                                                    semi_test_dataloader)

    f1score, accuracy, report = train_test_svc(labled_features, true_labels, 
                                               unlabeled_features, original_labels, 
                                               class_names, image_path)

    print(f"\n For {dataset_name} dataset f1_score : {f1score:.5f} \n accuracy : {accuracy:.5f}\n Classification report : \n {report}")

    p_at_1 = 0
    p_at_5 = 0
    p_at_10 = 0

    all_retrieved_labels = []
    all_original_labels = [] 

    for i, feature in enumerate(labled_features):
        indexes = find_closest_vector_indexes(feature, unlabeled_features, top_n = 10)
        retrieved_image_labels = []
        for j in range(len(original_labels)):
            if j in indexes:
                retrieved_image_labels.append(original_labels[j].item())
        original_image_label = true_labels[i]
        #print(f"feature lable : {original_image_label}, retrieved labels : {retrieved_image_labels} ")

        p_at_1 = p_at_1 + precision_at_k(retrieved_image_labels, original_image_label, 1)
        p_at_5 = p_at_5 + precision_at_k(retrieved_image_labels, original_image_label, 5)
        p_at_10 = p_at_10 + precision_at_k(retrieved_image_labels, original_image_label, 10)

        all_retrieved_labels.append(retrieved_image_labels)
        all_original_labels.append(original_image_label)

    print(f"P@1 : {p_at_1/len(labled_features):.5f}")
    print(f"P@5 : {p_at_5/len(labled_features):.5f}")
    print(f"P@10: {p_at_10/len(labled_features):.5f}")

    map_score = mean_average_precision(all_retrieved_labels, all_original_labels)
    mrr_score = mean_reciprocal_rank(all_retrieved_labels, all_original_labels)

    print(f"Mean Average Precision : {map_score:.5f}")
    print(f"Mean Reciprocal Rank   : {mrr_score:.5f}")


