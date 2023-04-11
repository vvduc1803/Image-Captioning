# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import os
import torch
import config
import shutil
import argparse
import numpy as np

from model import ImgCaption_Model
from dataset import get_loader
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import plot_examples, save_checkpoint, load_checkpoint_to_continue

# Initialize arguments and parse the arguments for easy run in terminal
def get_args():
    parse = argparse.ArgumentParser(description='Train Image Captioning Model')
    parse.add_argument('--epochs', '-e', type=int, default=config.epochs, help='number of epochs')
    parse.add_argument('--device', '-d', type=str, default=config.device, help='device to training')
    parse.add_argument('--load_model', type=bool, default=config.load_model, help='load model')
    parse.add_argument('--save_model', type=bool, default=config.save_model, help='save model')
    parse.add_argument('--lr', type=float, default=config.lr, help='learning rate of model')
    parse.add_argument('--batch-size', '-b', type=int, default=config.batch_size, help='number of batch size')
    parse.add_argument('--log-path', '-l', type=str, default=config.log_path, help='number of batch size')
    parse.add_argument('--save-path', '-s', type=str, default=config.save_path, help='number of batch size')
    parse.add_argument('--train', type=str, default=config.train, help='path to train dataset')
    parse.add_argument('--val', type=str, default=config.val, help='path to validation dataset')
    parse.add_argument('--test', type=str, default=config.test, help='path to test dataset')
    parse.add_argument('--transform', default=config.transform, help='Compose transform of images')
    parse.add_argument('--embed-size', default=config.embed_size, help='Size of embedding')
    parse.add_argument('--hidden-size', default=config.hidden_size, help='Number of hidden nodes in RNN')
    parse.add_argument('--num-layer', default=config.num_layer, help='Number of layers lstm stack')
    parse.add_argument('--num-workers', default=config.num_workers, help='Number of core CPU use to load data')
    args = parse.parse_args()
    return args

def train_fn(model, criterion, optimizer, train_loader, epoch, writer, device):
    # Set model to train mode
    model.train()

    # Initialize loss
    train_loss = []

    # Use tqdm for visual training process
    progress_bar = tqdm(train_loader, colour="green")

    # Load over iteration
    for idx, (images, captions) in enumerate(progress_bar):
        # Convert to device
        images = images.to(device)
        captions = captions.to(device)

        # Predict captions
        predicts = model(images, captions[:-1])

        # Compute loss
        loss = criterion(predicts.reshape(-1, predicts.shape[2]), captions.reshape(-1))
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # For visual
        progress_bar.set_description(
            f"Epoch {epoch + 1}: Iteration {idx + 1}/{len(train_loader)} Loss {np.mean(train_loss):.3f}")
        writer.add_scalar("Train/Loss", np.mean(train_loss), idx + epoch * len(train_loader))

def val_fn(model, criterion, val_loader, writer, epoch, device):
    # Set model to eval mode
    model.eval()

    # Initialize validation loss
    val_loss = []

    # For visual
    progress_bar = tqdm(val_loader, colour="green")

    with torch.inference_mode():
        # Load over iteration
        for idx, (images, captions) in enumerate(progress_bar):
            # Convert to device
            images = images.to(device)
            captions = captions.to(device)

            # Predict captions
            predicts = model(images, captions[:-1])

            # Compute loss
            loss = criterion(predicts.reshape(-1, predicts.shape[2]), captions.reshape(-1))
            val_loss.append(loss.item())

            # For visual
            progress_bar.set_description(
                f"Epoch {epoch + 1}: Iteration {idx + 1}/{len(val_loader)} Loss {np.mean(val_loss):.3f}")
        writer.add_scalar("Val/Loss", np.mean(val_loss), epoch)

    return np.mean(val_loss)


def train():
    # Initialize arguments for easy run in terminal
    args = get_args()

    # Load data
    train_dataset, train_loader = get_loader(root_folder=args.train,
                                             transform=args.transform,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=True)

    val_dataset, val_loader = get_loader(root_folder=args.val,
                                         transform=args.transform,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=True)

    # Load model, criterion and optimizer
    model = ImgCaption_Model(args.embed_size,
                             args.hidden_size,
                             len(train_dataset.vocab),
                             args.num_layer).to(args.device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Create tensorboard for view training process
    writer = SummaryWriter(args.log_path)

    # Start epoch and start loss
    start = 0
    best_loss = 1000

    # Load model and create log and save directory
    if args.load_model:
        model, start = load_checkpoint_to_continue(args.save_path, model, optimizer, args.lr, args.device)
    else:
        if os.path.isdir(args.log_path):
            shutil.rmtree(args.log_path)
        os.mkdir(args.log_path)
        if os.path.isdir(args.save_path):
            shutil.rmtree(args.save_path)
        os.mkdir(args.save_path)

    # Training process
    for epoch in range(start, args.epochs):
        train_fn(model, criterion, optimizer, train_loader, epoch,  writer,args.device)
        last_loss = val_fn(model, criterion, val_loader, writer, epoch, args.device)

        # Save model
        if args.save_model:
            best_loss = save_checkpoint(model, optimizer, epoch, args.save_path, last_loss, best_loss)

        # Plot some examples
        plot_examples(model, args.device, val_dataset, train_dataset.vocab)

        print('-------------------------------------------------')

if __name__ == '__main__':
    train()