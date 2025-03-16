import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from autoencoder import Autoencoder, train_autoencoder, extract_latent_vectors
from kmeans import Kmeansclustering
from evaluation import evaluate_silhouette, evaluate_dunn_index
import numpy as np

# Load CIFAR-10 dataset with proper transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load training and test datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Setup device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the Autoencoder
autoencoder = train_autoencoder(trainloader, device)

# Extract latent space representations from the test dataset
latent_TEST_DATA = extract_latent_vectors(autoencoder, testloader, device)

# Evaluate clustering using Silhouette Score
cluster_range = [8, 9, 10, 11, 12]
evaluate_silhouette(latent_TEST_DATA, cluster_range)

# Evaluate clustering using Dunn Index
evaluate_dunn_index(latent_TEST_DATA, cluster_range)