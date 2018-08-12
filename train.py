"""
Mirko Muscara
train.py
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
    parser = argparse.ArgumentParser(description='Get arguments for Neural Network training:')
    parser.add_argument('data_dir', type=str, help='mandatory data directory')
    parser.add_argument('--save_dir', default='', help='Directory to save checkpoint.')
    parser.add_argument('--pixels_frame', default=224, help='Pixel size of crop.')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='default learning rate' )
    parser.add_argument('--hidden_units', default=[25088, 4096, 1024], type=str, help='default hidden layer sizes')
    parser.add_argument('--output_size', default=102, type=int, help='default output_size')
    parser.add_argument('--drop_rate', default=0.5, type=int, help='default drop rate')
    parser.add_argument('--epochs', default=10, type=int, help='default training epochs')
    
    return parser.parse_args()

def get_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    valid_test_transforms = transforms.Compose([transforms.Resize(256), 
                                          transforms.CenterCrop(pixels),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, 
                                                               norm_std)])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(pixels),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(norm_mean,
                                                                norm_std)])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms) 
    validate_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    #Grab class ids
    class_idx = train_dataset.class_to_idx
    
    return train_loader, valid_loader, test_loader, class_idx, train_transforms, valid_test_transforms

def get_pretrain_model():
    return models.vgg19(pretrained=True)

def rebuild_model(model, hidden_layers, output_size, learning_rate, drop_p):
        
    for par in model.parameters():
        par.requires_grad = False
 
    classifier = nn.Sequential(OrderedDict([('input',  nn.Linear(hidden_layers[0], hidden_layers[1])),
                                            ('relu1',  nn.ReLU()),
                                            ('dropout1',  nn.Dropout(drop_p)),
                                            ('linear2',  nn.Linear(hidden_layers[1], hidden_layers[2])),
                                            ('relu2',  nn.ReLU()),
                                            ('dropout2',  nn.Dropout(drop_p)),
                                            ('linear3', nn.Linear(hidden_layers[2], output_size)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=10):
    steps = 0
    cuda = torch.cuda.is_available()
    print_every = 40
    
    if cuda:
        print('GPU TRAINING')
        model.cuda()
    else:
        print('CPU TRAINING')
        model.cpu()

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs), Variable(labels)
            steps += 1
            
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
            
                accuracy = 0
                validation_loss = 0
            
                for ii, (images, labels) in enumerate(valid_loader):
            
                    with torch.no_grad():
                        inputs = Variable(images)
                        labels = Variable(labels)
        
                        if cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()
        
                        output = model.forward(inputs)
                        validation_loss += criterion(output, labels).item()
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {} / {}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

                running_loss = 0
                model.train()
                
    print('{} epochs complete. Model trained.'.format(epochs))
    
    return model

def validate_model(model, test_loader):
    cuda = torch.cuda.is_available()
    
    if cuda:
        print('GPU TESTING')
        model.cuda()
    else:
        print('CPU TESTING')
              
    model.eval()
              
    acc = 0
    test_loss = 0

    #Forward pass
    for images, labels in iter(test_loader):
        
        with torch.no_grad():
            images, labels = Variable(images), Variable(labels)
        
            if cuda:
                images, labels = images.cuda(), labels.cuda()
        
            out = model.forward(images)
            test_loss += criterion(out, labels).data.item()

            ps = torch.exp(out).data
            equality = (labels.data == ps.max(1)[1])

            acc += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
          "Test Accuracy: {:.3f}".format(acc/len(test_loader)))

def save_chackpoint(model, input_size, output_size, epochs, train_transforms, learning_rate, class_idx, optimizer):
    save_dir = os.getcwd()

    saved_model = {
        'input_size':input_size,
        'output_size': output_size,
        'epochs':epochs,
        'batch_size': 64,
        'data_transforms': train_transforms,
        'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True],
        'model': model,
        'learning_rate': learning_rate,
        'class_to_idx': class_idx,
        'optimizer': optimizer.state_dict(),
        'classifier': model.classifier,
        'state_dict': model.state_dict() 
        }

    #Save checkpoint in current directory unless otherwise specified by save_dir
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
    torch.save(saved_model, save_path)
    print('Model saved at {}'.format(save_path))

def main():
    args = get_input_args()
    train_loader, valid_loader, test_loader, class_idx, train_transforms, valid_test_transforms = get_loaders(args.data_dir)
    model = get_pretrain_model()
    input_size = (args.pixels_frame**2)/2
    model, criterion, optimizer = rebuild_model(model, args.hidden_units, args.output_size, args.learning_rate, args.drop_rate)
    trained_model = train_model(model, criterion, optimizer, train_loader, valid_loader, args.epochs)
    test = validate_model(model, test_loader)
    save_chackpoint(args.save_dir, model, input_size, args.output_size, args.epochs, train_transforms, args.learning_rate, class_idx, optimizer)
    pass

if __name__ == '__main__':
    main()