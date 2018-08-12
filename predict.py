"""
Mirko Muscara
predict.py
"""
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import argparse
import os
import torch
import numpy as np
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict

def get_input_args():
    parser = argparse.ArgumentParser(description='Get arguments for Neural Network prediction:')
    parser.add_argument('input', type=str, help='image to process and predict')
    parser.add_argument('checkpoint', type=str, help='cnn to load')
    parser.add_argument('--top_k', default=1, type=int, help='default top_k results')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image_ratio = image.size[1] / image.size[0]
    image = image.resize((256, int(image_ratio*256)))
    # left, upper, right, lower
    #image = image.crop((16, 16, 224, 224))
    half_the_width = image.size[0] / 2
    half_the_height = image.size[1] / 2
    image = image.crop((half_the_width - 112,
                       half_the_height - 112,
                       half_the_width + 112,
                       half_the_height + 112))
    
    image = np.array(image)
    image = image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std_dev
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def set_idx_to_class(model):
    model.idx_to_class = dict([[v, k] for k, v in model.class_to_idx.items()])
    return model

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
        
    # TODO: Implement the code to predict the class from an image file
    image = None
    model.eval()
    with Image.open(image_path) as img:
        image = process_image(img)
        
    if torch.cuda.is_available():
        image = image.cuda()
    
    with torch.no_grad():
        image = Variable(image.unsqueeze(0))
        output = model.forward(image.float())
        ps = torch.exp(output)
        prob, idx = ps.topk(topk)
    probs = [y for y in np.array(prob.data[0])]
    classes = [model.idx_to_class[x] for x in np.array(idx.data[0])]
    return probs, classes

def main():
    args = get_input_args()
    model = load_checkpoint(args.checkpoint)
    norm_image = process_image(in_args.input)
    model = set_idx_to_class(model)
    probs, classes = predict(norm_image, model, in_args.top_k)
    print_predict(classes, probs)
    pass

if __name__ == '__main__':
    main()