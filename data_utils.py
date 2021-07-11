import matplotlib.pyplot as plt

import torch 
from torchvision import datasets, transforms

from PIL import Image
import numpy as np

from config import *

'''
Our suggestion is to create a file just for functions and classes relating to the model 
and another one for utility functions like loading data and preprocessing images. 
'''

def swap(class_to_idx):
    return {v: k for (k, v) in class_to_idx.items()}

def load_data(data_dir):
    # data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # : Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
                                transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(CROP_DIM),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(MEANS,SDS)
    ])
    valid_transforms = transforms.Compose([
                                transforms.Resize(RESIZE_DIM),
                                transforms.CenterCrop(CROP_DIM),
                                transforms.ToTensor(),
                                transforms.Normalize(MEANS,SDS)
    ])
    test_transforms = transforms.Compose([
                                transforms.Resize(RESIZE_DIM),
                                transforms.CenterCrop(CROP_DIM),
                                transforms.ToTensor(),
                                transforms.Normalize(MEANS,SDS)
    ])


    # : Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

    # : Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True)

    return train_loader, valid_loader, test_loader, train_data.class_to_idx
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # : Process a PIL image for use in a PyTorch model
   
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    # Provide the target width and height of the image
    img = Image.open(image,'r')
    
    w,h = img.size

    if w>h:
        img = img.resize((round(256*w/h),256))
    else:
        img = img.resize((256,round(256*h/w)))

    # Crop out the center 224x224 portion of the image
    resized_w, resized_h = img.size

    # The crop method from the Image module takes four coordinates as input.
    # The right can also be represented as (left+width)
    # and lower can be represented as (upper+height).

    left = (resized_w - 224)/2
    upper = (resized_h - 224)/2
    right = left + 224
    bottom = upper + 224

    coord = (round(left),round(upper),round(right),round(bottom))

    cropped = img.crop(coord)

    # Convert color channels to 0-1
    np_img = np.asarray(cropped)/255

    # Normalize the image
    np_img =  (np_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # Reorder dimensions
    np_img = np_img.transpose((2, 0, 1))
    
    return torch.FloatTensor(np_img)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    # image = image.numpy().transpose((1, 2, 0))
    # avoid array - Torch - array conversion
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    if title is not None:
        ax.set_title(title)
        
    return ax