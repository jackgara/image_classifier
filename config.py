

PRINT_EVERY = 20
BATCH_SIZE = 64
MEANS = [0.485, 0.456, 0.406]
SDS = [0.229, 0.224, 0.225]
CROP_DIM = 224
RESIZE_DIM = 256
CAT_TO_NAME_FILE= 'cat_to_name.json'


#optional default values
GPU = False
ARCHITECTURE='vgg19'
DEVICE='cpu'
EPOCHS = 5
LEARN_RATE = 0.001
HIDDEN_UNITS = 4096
SAVE_DIRECTORY = 'checkpoints'
TOP_K=5