from workspace_utils import active_session

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import models

import numpy as np
import json

from config import *
from data_utils import process_image, imshow


'''
set device
'''


def set_device(gpu):
    return torch.device(
        "cuda" if (gpu and torch.cuda.is_available()) else "cpu")


'''
Create the Model
'''


def init(gpu, arch):

    # Use GPU if it's available
    device = set_device(gpu)

    # Load a pre-trained network
    model = getattr(models, arch)(pretrained=True)

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False

    print(' Model Init ...........' )
    print(' Architecture : %s' % arch )
    print(' Device : %s' % device )
    print(' ......................DONE \n')

    return model, device


'''
Setup Model 
'''


def setup(model, device, hidden_units, learning_rate):

    with open(CAT_TO_NAME_FILE, 'r') as f:
        cat_to_name = json.load(f)

    features_size = model.classifier._modules['0'].in_features
    classifier = nn.Sequential(
        nn.Linear(features_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_units, len(cat_to_name)),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # model.to(device)

    print(' Model Setup ...........' )
    print(' Classifier : ' )
    print('     Categories : %s' % len(cat_to_name) )
    print('     Features : %s' % features_size )
    print('     Hidden Units : %s' % hidden_units )
    print(' Optimizer:' )
    print('     Learning Rate : %s' % learning_rate )
    print(' ........................DONE \n')
    
    return model, criterion, optimizer, device


'''
 Predict the class (or classes) of an image using a trained deep learning model.
'''


def predict(device, image_path, model, id_to_class, cat_to_name, topk=5):
    # : Implement the code to predict the class from an image file
    model.to(device)
    model.eval()

    # get the torch image and add a dimension [] to match model inputs shape
    tensor_img = torch.unsqueeze(process_image(image_path), 0)

    with torch.no_grad():
        tensor_img = tensor_img.to(device)
        # forward pass
        output = model.forward(tensor_img)

    # get proba
    probs = torch.exp(output)
    ps, classes = probs.topk(topk)

    classes = label_names(classes, id_to_class, cat_to_name)

    return ps.cpu().numpy()[0], classes


def label_names(labels, id_to_class,cat_to_name):
    names_array = np.array([])
    for i in np.nditer(labels):
        names_array = np.append(names_array, cat_to_name[id_to_class[int(i)]])
    return names_array


'''
Predict and display 5 more likely classes
'''


def display_probabilities(device, image_path, model, id_to_class, cat_to_name):

    # get real class name
    class_idx = image_path.split('/')[-2]
    class_name = cat_to_name.get(class_idx)

    # predict classes and probs
    probs, classes = predict(device, image_path, model, id_to_class, topk=5)

    # plot

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 10), ncols=2)
    ax1 = plt.subplot(2, 1, 1)

    # Plot flower
    imshow(process_image(image_path), ax1, class_name)

    # Plot probs
    ax2 = fig.add_subplot(2, 1, 2)
    y_pos = np.arange(len(classes))

    ax2.barh(y_pos, probs, align='center', color='blue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(label_names(classes, id_to_class))
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probability')
    plt.tight_layout()

'''
Load the architecture from a checkpoint
'''
def load_checkpoint_arch(filepath):
    checkpoint = torch.load(filepath)
    return checkpoint['architecture']

'''
Load a model from a checkpoint
'''


def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.class_to_name_file = checkpoint['class_to_name_file']
    model.load_state_dict(checkpoint['state_dict'])
    lr = checkpoint['learning_rate']

    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    return optimizer, model


'''
Save a model to a checkpoint
'''


def save_checkpoint(save_directory, architecture, model, optimizer, class_to_idx, class_to_name_file, learn_rate, epochs):
    checkpoint = {'architecture' : architecture,
                 'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': class_to_idx,
                  'class_to_name_file': class_to_name_file,
                  'learning_rate': learn_rate,
                  'epochs': epochs
                  }
    print('Saving checkpoint into %s' % (save_directory+'/'+'checkpoint.pth'))
    torch.save(checkpoint, save_directory+'/'+'checkpoint.pth')
    print('.......................DONE')

'''
Test data
'''


def test(device, model, loader, criterion):
    model.eval()
    with torch.no_grad():
        loss, accuracy = validation(device, model, loader, criterion)

        print("........................................")
        print(
            # avg loss for all batches in valid_loader
            f"Loss: {loss/len(loader):.3f}.. "
            # avg accuracy fpr all batches in valid_loader
            "Accuracy: %d%%" % ((accuracy/len(loader))*100)
        )


'''
Train the classifier layers using backpropagation 
using the pre-trained network to get the features
'''


def train(model, train_loader, valid_loader, epochs, print_every,
          criterion, optimizer, device):

    model.to(device)

    with active_session():

        step = 0
        running_loss = 0

        batches_total = len(train_loader.batch_sampler)*epochs
        batch_size = train_loader.batch_size

        print("Training on %s" % device)
        print("%s Epochs " % epochs)
        print("%s Batches of %s Images per batch" %
              (batches_total, batch_size))
        print("Average loss and accuracy displayed every %s batches\n" %
              print_every)

        for epoch in range(epochs):
            # iterate trough batches
            for inputs, labels in train_loader:
                # model to train by default
                # model.train()

                step += 1

                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                # reset optimizer grads with parenthesis !! () -> should give error
                optimizer.zero_grad()

                # forward pass
                logps = model.forward(inputs)
                # error
                loss = criterion(logps, labels)

                # backward pass
                loss.backward()
                # optimizer updates weights
                optimizer.step()

                # update accumulated loss
                running_loss += loss.item()

                # Track the loss and accuracy on the validation set to determine the best hyperparameters
                if step % PRINT_EVERY == 0:

                    # model to evaluation ( turn off Dropout )
                    model.eval()

                    # turn off parameter updating, so speed up the process
                    with torch.no_grad():
                        valid_loss, accuracy = validation(device,
                                                          model, valid_loader, criterion)

                    print("........................................")
                    print(f"Epoch {epoch+1}/{epochs}...............")
                    print(
                        f"Batch : {step}, Images {batch_size*step} - {batch_size*(step+1)} .......")
                    print("........................................")
                    print(
                        # avg running_loss for last [print_every] batches
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        # avg loss for all batches in valid_loader
                        f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                        # avg accuracy fpr all batches in valid_loader
                        "Validation accuracy: %d%%" % (
                            (accuracy/len(valid_loader))*100)
                    )

                    # reset running_loss for next batch
                    running_loss = 0
                    # model to train ( turn on Dropout )
                    model.train()


'''
Validation function for validation and test datasets
'''


def validation(device, model, valid_loader, criterion):
    valid_loss = 0
    accuracy = 0

    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        # forward pass
        output = model.forward(inputs)
        # error
        valid_loss += criterion(output, labels).item()
        # get proba
        ps = torch.exp(output)
        # measure matching
        equality = (labels.data == ps.max(dim=1)[1])
        # calculate accuracy
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy
