from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from utils import read_images, get_labels


class CustomMNISTDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class CustomRetinaDataset(Dataset):
    def __init__(self, data, labels, train=None):
        self.data = data
        self.labels = labels
        self.train = train
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(degrees=15),
            #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        #if self.train == True:
        #    image = self.transform(image)
        return image, label

def get_images_labels(dataset):
    images = []
    labels = []
    for data in dataset:
        image, label = data
        images.append(image)
        labels.append(label)
    return images, labels

def get_mnist_data():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)
    return train_dataset, test_dataset

def get_fashion_mnist_data():
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True)
    return train_dataset, test_dataset

def get_cifar10_data():
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    return train_dataset, test_dataset

def get_retina_dataset():
    train_images = read_images("medical_data/Training_Set/Training_Set/Training/")
    test_images = read_images("medical_data/Test_Set/Test_Set/Test/")
    validation_images = read_images("medical_data/Evaluation_Set/Evaluation_Set/Validation/")

    _, train_labels = get_labels("medical_data/Training_Set/Training_Set/RFMiD_Training_Labels.csv")
    label_names, test_labels = get_labels("medical_data/Test_Set/Test_Set/RFMiD_Testing_Labels.csv")
    _, validation_labels = get_labels("medical_data/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv")

    return train_images, validation_images, test_images, train_labels, validation_labels, test_labels, label_names

def process_dataset(images, labels, label_names):
    selected_labels = []
    selected_images = []
    selected_label_names = []
    class_selected = []
    for i in range(46):
        indices = []
        sum = 0
        for index, label in enumerate(labels):
            if label==i:
                sum = sum + 1
        if sum > 5:
            class_selected.append(i)

    for i in class_selected:
        selected_label_names.append(label_names[i])
    
    for i in range(len(labels)):
        if labels[i] in class_selected:
            selected_labels.append(class_selected.index(labels[i]))
            selected_images.append(images[i])

    return selected_images, selected_labels, selected_label_names

def get_retina_dataloaders(batch_size, test_size, seed):
    train_images, validation_images, test_images, train_labels, validation_labels, test_labels, label_names = get_retina_dataset()
    test_images.extend(validation_images)
    test_labels.extend(validation_labels)
    test_images, test_labels, label_names = process_dataset(test_images, test_labels, label_names)
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2, 
                                                                                                random_state=seed)
    semi_train_images, semi_test_images, semi_train_labels, semi_test_labels = train_test_split(test_images, test_labels, test_size=test_size, 
                                                                                                random_state=seed)
    
    train_dataset = CustomRetinaDataset(data=train_images, labels=train_labels, train=True)
    validation_dataset = CustomRetinaDataset(data=validation_images, labels=validation_labels, train=False)
    semi_train_dataset = CustomRetinaDataset(data=semi_train_images, labels=semi_train_labels, train=False)
    semi_test_dataset = CustomRetinaDataset(data=semi_test_images, labels=semi_test_labels, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    semi_train_dataloader = DataLoader(semi_train_dataset, batch_size=batch_size, shuffle=True)
    semi_test_dataloader = DataLoader(semi_test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, semi_train_dataloader, semi_test_dataloader, label_names
    
def get_dataloaders(batch_size, test_size, seed, dataset_name):
    if dataset_name == "mnist":
        train_dataset, test_dataset = get_mnist_data()
    elif dataset_name == "fashion_mnist":
        train_dataset, test_dataset = get_fashion_mnist_data()
    elif dataset_name == "cifar10":
        train_dataset, test_dataset = get_cifar10_data()

    train_images, train_labels = get_images_labels(train_dataset)
    test_images, test_labels = get_images_labels(test_dataset)

    train_images, val_images, train_labels, val_labels = get_train_val_data(train_images, train_labels, seed)
    semi_train_images, test_images, semi_train_labels, test_labels = get_semi_supervised_data(test_images, 
                                                                                              test_labels, 
                                                                                              test_size=test_size,
                                                                                              seed = seed)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    cifar10_ransform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == "mnist" or dataset_name == "fashion_mnist":
        mnist_train_dataset = CustomMNISTDataset(data=train_images, labels=train_labels, transform=transform)
        mnist_val_dataset = CustomMNISTDataset(data=val_images, labels=val_labels, transform=transform)
        mnist_semi_train_dataset = CustomMNISTDataset(data=semi_train_images, labels = semi_train_labels, transform=transform)
        mnist_semi_test_dataset = CustomMNISTDataset(data=test_images, labels = test_labels,  transform=transform)

    else:
        mnist_train_dataset = CustomCIFAR10Dataset(data=train_images, labels=train_labels, transform=cifar10_ransform)
        mnist_val_dataset = CustomCIFAR10Dataset(data=val_images, labels=val_labels, transform=cifar10_ransform)
        mnist_semi_train_dataset = CustomCIFAR10Dataset(data=semi_train_images, labels = semi_train_labels, transform=cifar10_ransform)
        mnist_semi_test_dataset = CustomCIFAR10Dataset(data=test_images, labels = test_labels,  transform=cifar10_ransform)


    train_dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(mnist_val_dataset, batch_size=batch_size, shuffle=True)
    semi_train_dataloader = DataLoader(mnist_semi_train_dataset, batch_size=batch_size, shuffle=True)
    semi_test_dataloader = DataLoader(mnist_semi_test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, semi_train_dataloader, semi_test_dataloader

def get_train_val_data(images, labels, seed=42):
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=seed, stratify=labels)
    return X_train, X_val, y_train, y_val

def get_semi_supervised_data(images, labels, test_size, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=seed, stratify=labels)
    return X_train, X_test, y_train, y_test
