"""
This script contains code to download and import training/test
datasets, and code to preprocess the data ready for training.
"""

##################################################################

# Specific Imports

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

##################################################################

def import_data(data):
    """Function to import train and test datasets."""

    if data == 'train':
        train=True
    elif data == 'test':
        train=False
    else:
        print("""Please specify data to be either 'test' or 'train'""")
        return

    data = datasets.FashionMNIST(
        root='data',
        train=train,
        transform=ToTensor(),
        download=True
    )

    return data

def generate_loader(data):
    """Function to segment data into batches, which are reshuffled 
    at every epoch to reduce model overfitting (and which also enable
    multiprocessing)."""

    loader = DataLoader(
            data,
            batch_size=100,
            shuffle=True)

    return loader