import argparse

from config import *
from data_utils import load_data
from model_utils import init, setup, train, save_checkpoint

parser = argparse.ArgumentParser(
    description='Train a new network on a data set'
)

# required argument

parser.add_argument('data_directory', action='store',
                    type=str,
                    help='Image Data Directory to train')

# optionals

parser.add_argument('-s', '--save_dir', action='store', type=str, dest='save_directory',
                    help='Directory to save Checkpoints')

parser.add_argument('-a', '--arch ', choices=['vgg19', 'vgg16'], action='store',  type=str, dest='arch', default='vgg19',
                    help='Architecture')

parser.add_argument('-l', '--learning_rate ', action='store', type=float, dest='learning_rate',
                    help='Learning Rate')

parser.add_argument('-e', '--epochs ', action='store',  type=int, dest='epochs',
                    help='Epochs')

parser.add_argument('-hu', '--hidden_units ', action='store', type=int, dest='hidden_units',
                    help='Number of Hidden Units')

parser.add_argument('-g', '--gpu ', action='store_false', default='True', dest='gpu',
                    help='Use GPU device')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')


args = parser.parse_args()

'''
Train a new network on a dataset and save the model as a checkpoint
'''
print('Training .................... \n')

# Load args
data_directory = args.data_directory

# Load arg options
save_directory = args.save_directory or SAVE_DIRECTORY
gpu = args.gpu or GPU
architecture = args.arch or ARCHITECTURE
learning_rate = args.learning_rate or LEARN_RATE
epochs = args.epochs or EPOCHS
hidden_units = args.hidden_units or HIDDEN_UNITS


'''SETUP'''

''' Create the Model'''
model, device = init(gpu, architecture)

''' Setup the Model to Train'''
model, criterion, optimizer, device = setup(
    model, device, hidden_units, learning_rate)


''' Getting the Data'''
train_loader, valid_loader, test_loader, class_to_idx = load_data(
    data_directory)


''' Train '''
train(model, train_loader, valid_loader, epochs,
      PRINT_EVERY, criterion, optimizer, device)

''' Save Checkpoint'''
save_checkpoint(save_directory, architecture, model, optimizer,
                class_to_idx, CAT_TO_NAME_FILE, learning_rate, epochs)

'''
python train.py flowers
'''

'''
python train.py -s checkpoints_hyperparam --arch vgg16 -l 0.005 -e 1 -hu 1024 flowers
'''