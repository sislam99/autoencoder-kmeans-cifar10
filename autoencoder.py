import torch  
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

class Autoencoder(nn.Module):
    """A simple Convolutional Autoencoder"""
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder part - compresses image data
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Decoder part - reconstructs image from compressed features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(trainloader, device, num_epochs=100):
    autoencoder = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.5f}')

    return autoencoder

def extract_latent_vectors(autoencoder, testloader, device):
    latent_vectors = []
    autoencoder.eval()
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            latent_repr = autoencoder.encoder(images)
            latent_repr = latent_repr.view(latent_repr.size(0), -1)
            latent_vectors.append(latent_repr.cpu().numpy())
    return np.concatenate(latent_vectors, axis=0)
