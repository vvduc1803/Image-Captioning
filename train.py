import torch
import config
import argparse

from utils import print_examples, save_checkpoint, load_checkpoint

# Initialize arguments and parse the arguments
def get_args():
    parse = argparse.ArgumentParser(description='Train Image Captioning Model')
    parse.add_argument('--epochs', '-e', type=int, default=config.epochs, help='number of epochs')
    parse.add_argument('--learning-rate', '-lr', type=float, default=config.lr, help='learning rate of model')
    parse.add_argument('--batch-size', '-b', type=int, default=config.batch_size, help='number of batch size')
    parse.add_argument('--transform', default=config.transform, help='Compose transform of images')
    parse.add_argument('--embed-size', default=config.embed_size, help='Size of embedding')
    parse.add_argument('--hidden-size', default=config.hidden_size, help='Number of hidden nodes in RNN')
    parse.add_argument('--num-layer', default=config.num_layer, help='Number of layers lstm stack')
    parse.add_argument('--num-worker', default=config.num_worker, help='Number of core CPU use to load data')

def train_fn():
    pass

def val_fn():
    pass

def train():
    pass

if __name__ == '__main__':
    train()