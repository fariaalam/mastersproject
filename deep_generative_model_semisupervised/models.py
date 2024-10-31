import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_Mnist(nn.Module):
    def __init__(self):
        super(Encoder_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7)                     
        self.flatten = nn.Flatten()                                       
        self.fc = nn.Linear(64, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Decoder_Mnist(nn.Module):
    def __init__(self):
        super(Decoder_Mnist, self).__init__()
        self.fc = nn.Linear(16, 64)
        self.unflatten = nn.Unflatten(1, (64, 1, 1))
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=7)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.unflatten(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

class MnistAutoencoder(nn.Module):
    def __init__(self, dataset_name = "mnist"):
        super(MnistAutoencoder, self).__init__()
        self.encoder = Encoder_Mnist()
        self.decoder = Decoder_Mnist()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class Encoder_FashionMnist(nn.Module):
    def __init__(self):
        super(Encoder_FashionMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7)                     
        self.flatten = nn.Flatten()                                       
        self.fc = nn.Linear(64, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Decoder_FashionMnist(nn.Module):
    def __init__(self):
        super(Decoder_FashionMnist, self).__init__()
        self.fc = nn.Linear(16, 64)
        self.unflatten = nn.Unflatten(1, (64, 1, 1))
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=7)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.unflatten(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

class FashionMnistAutoencoder(nn.Module):
    def __init__(self, dataset_name = "mnist"):
        super(FashionMnistAutoencoder, self).__init__()
        self.encoder = Encoder_FashionMnist()
        self.decoder = Decoder_FashionMnist()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder_CIFAR10(nn.Module):
    def __init__(self):
        super(Encoder_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(256*4*4, 64)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder_CIFAR10(nn.Module):
    def __init__(self):
        super(Decoder_CIFAR10, self).__init__()
        self.fc = nn.Linear(64, 256*4*4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x

class CIFAR10Autoencoder(nn.Module):
    def __init__(self):
        super(CIFAR10Autoencoder, self).__init__()
        self.encoder = Encoder_CIFAR10()
        self.decoder = Decoder_CIFAR10()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class RetinaEncoder(nn.Module):
    def __init__(self):
        super(RetinaEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.layers(x)


class RetinaDecoder(nn.Module):
    def __init__(self):
        super(RetinaDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 512 * 7 * 7),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Unflatten(1, (512, 7, 7)),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class RetinaAutoencoder(nn.Module):
    def __init__(self):
        super(RetinaAutoencoder, self).__init__()
        self.encoder = RetinaEncoder()
        self.decoder = RetinaDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
