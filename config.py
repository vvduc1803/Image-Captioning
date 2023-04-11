# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import torch

from torchvision import transforms

# Image and caption root
train = 'data/train'
val = 'data/val'
test = 'data/test'

# Image and caption path
captions = 'captions.json'
images = 'image'

# Log path and save path
log_path = 'logdir'
save_path = 'savedir'

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# All Parameters you can tune
lr = 3e-4
epochs = 100
embed_size = 256
hidden_size = 256
num_layer = 1
num_workers = 4
batch_size = 16

# Save, load model
load_model = False
save_model = True

# Compose transform image
transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomGrayscale(0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ]
    )