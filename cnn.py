import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18


IMAGE_WIDTH = 32
NUM_COLORS = 3
NUM_CLASSES = 8
NUM_EPOCHS = 3
BATCH_SIZE = 64
PATH = "resnet18.pt"


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize dataset. Note that transform and target_transform
        correspond to data transformations for train and test respectively.
        """
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.empty((0, IMAGE_WIDTH, IMAGE_WIDTH, NUM_COLORS), dtype=np.uint8)
        self.labels = []
        for data_file in data_files:
            dict = unpickle(data_file)
            self.labels += dict[b'labels']
            data = np.array(dict[b'data'])
            data = data.reshape(len(data), NUM_COLORS, IMAGE_WIDTH, IMAGE_WIDTH).transpose(0, 2, 3, 1)
            data.astype(np.uint8)
            self.images = np.concatenate((self.images, data))

    def __len__(self):
        """
        Return the length of dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Obtain a sample from dataset. 

        Parameters:
            x:      an integer, used to index into the data.

        Outputs:
            y:      a tuple (image, label)
        """
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    return CIFAR10(data_files, transform)


"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    return DataLoader(dataset, **loader_params)


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize the neural network.
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        self.model = resnet18()
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.fc.requires_grad_(True)
        self.model.fc.out_features = NUM_CLASSES

    def forward(self, x):
        """
        Perform a forward pass through neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        return self.model(x)


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    return torch.optim.Adam(params=model_params)


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train the neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """
    for x, y in train_dataloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y.long())
        model.zero_grad()
        loss.backward()
        optimizer.step()


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    Test the neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of the model and should not update the model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that is used to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on test images: {100 * correct // total} %')

    return correct / total

"""
7. Full model training and testing
"""
def run_model():
    """
    Outputs:
        model:              trained model
    """
    train_batches = ["cifar10_batches/data_batch_1", "cifar10_batches/data_batch_2", "cifar10_batches/data_batch_3", "cifar10_batches/data_batch_4", "cifar10_batches/data_batch_5"]
    transform = get_preprocess_transform("train")
    dataset = build_dataset(train_batches, transform)
    train_dataloader = build_dataloader(dataset, {"batch_size": BATCH_SIZE, "shuffle": True})
    model = build_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer("Adam", model.parameters(), {})

    train(train_dataloader, model, loss_fn, optimizer)

    test_batch = ["cifar10_batches/test_batch"]
    test_dataset = build_dataset(test_batch, transform)
    test_dataloader = build_dataloader(test_dataset, {"batch_size": BATCH_SIZE, "shuffle": False})
    test(test_dataloader, model)

    return model
    
