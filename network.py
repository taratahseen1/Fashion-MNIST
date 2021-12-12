"""
This script contains code to define the convolutional neural
network.
"""

##################################################################

# Specific Imports

import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss

from tqdm import tqdm

# Local Imports

from get_data import import_data, generate_loader

##################################################################

# Define Hyperparameters

NUM_EPOCHS = 10

##################################################################

class CNN(nn.Module):
    """This class defines the convolutional neural network to be 
    used with the Fashion-MNIST dataset."""

    def __init__(self) -> None:
        """Initialises initial module state."""
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(3,3),
                    stride=1,
                    padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )

        self.out = nn.Linear(
            in_features=32 * 14 * 14,
            out_features=10
        )

    def forward(self, x):
        """Function to feed forward input data x through the CNN."""
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output=self.out(x)
        return output

##################################################################

def initialise_training_environment():
    """Function to initialise optimisation function, loss function, and model."""

    # Define the model
    model = CNN()

    # Define the optimiser
    optimiser = optim.Adam(model.parameters(), lr=0.01) # lr = learning rate

    # Define loss function
    loss_func = CrossEntropyLoss()

    # Checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    print('Model Architecture:')
    print(model)

    return model, optimiser, loss_func

def train(model, optimiser, loss_func):
    """Function to train the CNN and return the trained CNN."""
    model.train()

    # Get training data
    train_data = import_data(data='train')

    # Get loader
    train_loader = generate_loader(train_data)

    # Train the model

    for epoch in tqdm(range(NUM_EPOCHS)):
        for batch_idx, (images, labels) in enumerate(train_loader):

            batch_x = Variable(images)
            batch_y = Variable(labels)

            # Clear gradients of the model parameters
            optimiser.zero_grad()
            
            # Get model output for this batch
            output = model(batch_x)

            # Find loss function for this batch
            loss_train = loss_func(output, batch_y)

            # Compute updated weights of all model parameters
            loss_train.backward()
            optimiser.step()

    return model

def test(model):
    """Function to test the trained CNN on the testing dataset."""

    model.eval()

    # Get test data
    test_data = import_data(data='test')

    # Get loader
    test_loader = generate_loader(test_data)

    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:

            # Get predictions on test images
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()

            # Count number of correct predictions
            correct += pred_y.eq(labels).sum()

        # # Calculate accuracy
        # accuracy = (
        #     correct / test_data.size()
        # )

    return images, labels, pred_y
        