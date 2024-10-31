import os 
import torch
import random 
import numpy as np
from train_evaluate import train_evaluate
from dataloader import get_dataloaders, get_retina_dataloaders
from models import MnistAutoencoder, CIFAR10Autoencoder, FashionMnistAutoencoder, RetinaAutoencoder
from train_evaluate import (
    find_closest_vector_indexes, 
    mean_reciprocal_rank, mean_average_precision, average_precision, precision_at_k, 
    find_closest_vector_indexes, cosine_similarity
)


if os.path.exists("saved_models/") == False:
    os.mkdir("saved_models/")   

if os.path.exists("confusion_matrix/") == False:
    os.mkdir("confusion_matrix/")   


#####################################################################
#              Mnist, Fashion Mnist, CIFAR10                        #
#####################################################################

epochs = 100
seed = 42
batch_size = 64 
test_size = 0.9
dataset_name = "cifar10"
model_path = "saved_models/"+dataset_name
image_path = "confusion_matrix/"+dataset_name+".png"
learning_rate  = 0.001

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

mnist_class_lables = ["zero", "one","two","three","four","five","six","seven","eight","nine"]
fashion_mnist_class_lables = ["T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
cifar10_class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class_names = None
if dataset_name == "fashion_mnist":
    class_names = fashion_mnist_class_lables
if dataset_name == "mnist":
    class_names = mnist_class_lables
elif dataset_name == "cifar10":
    class_names = cifar10_class_labels


if dataset_name == "fashion_mnist":
    model = FashionMnistAutoencoder()
elif dataset_name == "mnist":
    model = MnistAutoencoder()
elif dataset_name == "cifar10":
    model = CIFAR10Autoencoder()

train_dataloader, val_dataloader, semi_train_dataloader, semi_test_dataloader = get_dataloaders(
                                                                                            batch_size = batch_size, 
                                                                                            test_size = test_size, 
                                                                                            seed = seed,
                                                                                            dataset_name = dataset_name)



train_evaluate(epochs = epochs,
                model_path = model_path,
                batch_size = batch_size,
                test_size = test_size,
                seed = seed,
                dataset_name = dataset_name,
                train_dataloader = train_dataloader,
                val_dataloader = val_dataloader,
                semi_train_dataloader = semi_train_dataloader,
                semi_test_dataloader = semi_test_dataloader,
                model = model,
                learning_rate = learning_rate,
                class_names= class_names,
                image_path = image_path)

#####################################################################
#                  Retinal Dataset                                  #
#####################################################################
'''
epochs = 100
seed = 18
batch_size = 64 
test_size = 0.5 
dataset_name = "retina"
model_path = "saved_models/"+dataset_name
image_path = "confusion_matrix/"+dataset_name+".png"
learning_rate  = 0.001


torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


train_dataloader, val_dataloader, semi_train_dataloader, semi_test_dataloader, class_names = get_retina_dataloaders(
                                                                                            batch_size = batch_size, 
                                                                                            test_size = test_size, 
                                                                                            seed = seed)

                                                                                         
model = RetinaAutoencoder()
train_evaluate(epochs = epochs,
                model_path = model_path,
                batch_size = batch_size,
                test_size = test_size,
                seed = seed,
                dataset_name = dataset_name,
                train_dataloader = train_dataloader,
                val_dataloader = val_dataloader,
                semi_train_dataloader = semi_train_dataloader,
                semi_test_dataloader = semi_test_dataloader,
                model = model,
                learning_rate = learning_rate,
                class_names= class_names,
                image_path = image_path)
'''


#####################################################################
#                  For PCA + SVM                                    #
#####################################################################

print("\n\n Results With PCA \n\n")
import torch
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

train_data = []
train_labels = []
for data in semi_train_dataloader:
    images, labels = data
    for sample in images:
        flattened_list = sample.view(-1).tolist()
        train_data.append(flattened_list)
    for label in labels:
        train_labels.append(label.item())

pca = PCA(n_components=128)
pca.fit(train_data)
train_data = pca.transform(train_data)

test_data = []
test_labels = []
for data in semi_test_dataloader:
    images, labels = data
    for sample in images:
        flattened_list = sample.view(-1).tolist()
        test_data.append(flattened_list)
    for label in labels:
        test_labels.append(label.item())

test_data = pca.transform(test_data)

clf = SVC()
clf.fit(train_data, train_labels)
predictions = clf.predict(test_data)
f1score = f1_score(test_labels, predictions, average='micro')
accuracy = accuracy_score(test_labels, predictions)

print(f1score)

def information_retrieval(train_data, train_labels, test_data, test_labels):
    p_at_1 = 0
    p_at_5 = 0
    p_at_10 = 0

    all_retrieved_labels = []
    all_original_labels = [] 

    for i, feature in enumerate(train_data):
        indexes = find_closest_vector_indexes(feature, test_data, top_n = 10)
        retrieved_image_labels = []
        for j in range(len(test_labels)):
            if j in indexes:
                retrieved_image_labels.append(test_labels[j])
        original_image_label = train_labels[i]
        #print(f"feature lable : {original_image_label}, retrieved labels : {retrieved_image_labels} ")

        p_at_1 = p_at_1 + precision_at_k(retrieved_image_labels, original_image_label, 1)
        p_at_5 = p_at_5 + precision_at_k(retrieved_image_labels, original_image_label, 5)
        p_at_10 = p_at_10 + precision_at_k(retrieved_image_labels, original_image_label, 10)

        all_retrieved_labels.append(retrieved_image_labels)
        all_original_labels.append(original_image_label)

    print(f"P@1 : {p_at_1/len(train_data):.5f}")
    print(f"P@5 : {p_at_5/len(train_data):.5f}")
    print(f"P@10: {p_at_10/len(train_data):.5f}")

    map_score = mean_average_precision(all_retrieved_labels, all_original_labels)
    mrr_score = mean_reciprocal_rank(all_retrieved_labels, all_original_labels)

    print(f"Mean Average Precision : {map_score:.5f}")
    print(f"Mean Reciprocal Rank   : {mrr_score:.5f}")
    
information_retrieval(train_data, train_labels, test_data, test_labels)
