import argparse
import json

from config import *
from model_utils import predict, load_checkpoint_arch, load_checkpoint, init
from data_utils import swap

'''
Parser
'''
parser = argparse.ArgumentParser(
    description='Predict flower name from an image along with the probability of that name'
)

# required argument

parser.add_argument('image_path', action='store',
                    type=str,
                    help='Path to image to predict')

parser.add_argument('checkpoint', action='store',
                    type=str,
                    help='Checkpoint to use')

# optionals

parser.add_argument('-k', '--top_k', action='store', type=int, dest='top_k',
                    help='Top K classes to return')

parser.add_argument('-c', '--category_names ', action='store', type=str, dest='category_names',
                    help='Dictionary of Category Id to Name')

parser.add_argument('-g', '--gpu ', action='store_true', default='False', dest='gpu',
                    help='Use GPU for Inference')


parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')


args = parser.parse_args()


'''
 Uses a trained network to predict the class for an input image.
'''


# Load args
image_path = args.image_path
checkpoint = args.checkpoint

# options
top_k = args.top_k or TOP_K
gpu = args.gpu or GPU
cat_to_name_file = args.category_names or CAT_TO_NAME_FILE

'''
Create model 
'''
print('Creating the model ................')
# recreate the model with saved architecture
arch = load_checkpoint_arch(checkpoint)
model, device = init(gpu, arch)

'''
Load checkpoint 
'''
print('Loading the checkpoint ................')
# setup the model, classifier, optimizer
optimizer, model = load_checkpoint(checkpoint, model)


# setup cat-to-name dictionary
with open(cat_to_name_file, 'r') as f:
    cat_to_name = json.load(f)

# setup id-to-class dictionary
id_to_class = swap(model.class_to_idx)

'''
Predicts image class 
'''
print('Predicting  ................')
probs, classes = predict(device, image_path, model,
                         id_to_class, cat_to_name, top_k)

print('Probability ................ Class')
print('')
for p, c in zip(probs, classes):
    print("%.4f                     %s" % (p, c))

'''
Predict and display 5 more likely classes
'''
# display_probabilities(device, image_path, model, id_to_class)


'''
python predict.py flowers/test/100/image_07896.jpg checkpoints/checkpoint.pth
python predict.py flowers/test/11/image_03098.jpg checkpoints/checkpoint.pth
python predict.py flowers/test/12/image_03994.jpg checkpoints/checkpoint.pth
python predict.py flowers/test/13/image_05745.jpg  checkpoints/checkpoint.pth
python predict.py flowers/test/102/image_08004.jpg checkpoints/checkpoint.pth

'''
'''
python predict.py 'flowers/test/100/image_07896.jpg' checkpoints_hyperparam/checkpoint.pth -k 10 -c 'alt_cat_to_name.json' 
'''
# test with g False
'''
python predict.py 'flowers/test/100/image_07896.jpg' checkpoints_hyperparam/checkpoint.pth -k 10 -c 'alt_cat_to_name.json' -g

'''