# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import os
import torch
import random
import matplotlib.pyplot as plt

def read_caption(num_caption, vocab):
    """
    Convert caption form number to string
    Args:
        num_caption: caption form number
        vocab: vocabulary file
    Returns:
        A list of string (ex: [a, dog, in, the, sky])
    """
    str_caption = []
    for cap in num_caption[1:]:
        if vocab.itos[cap.item()] == "<EOS>":
            break
        str_caption.append(cap)

    return [vocab.itos[id.item()] for id in str_caption]

def plot_examples(model, device, dataset, vocab, num_examples=20):
    """
    Plot image, correct caption and predict caption of some image in dataset

    Args:
        model: pretrained-model to predict caption
        device: target device cpu and gpu
        dataset: dataset
        vocab: vocabulary
        num_examples: number examples plot

    Returns:
        Images of picture and caption
    """
    model.eval()
    model.to(device)

    # Load over examples
    for example in range(num_examples):
        # Take some example from dataset
        image, caption = dataset.__getitem__(random.randint(0, dataset.__len__()))
        image = image.to(device)

        # Print output
        correct = f"Example {example+1} CORRECT: " + " ".join(read_caption(caption, vocab))
        output = f"Example {example+1} OUTPUT: " + " ".join(model.caption_image(image.unsqueeze(0), vocab))
        print(correct)
        print(output)
        print('----------------------------------------------')

        # Plot image and caption
        fig, ax = plt.subplots()
        ax.imshow(dataset.img)
        ax.axis('off')
        fig.text(0.5, 0.05,
                 correct + '\n' + output,
                 ha="center")

        plt.show()

    model.train()


def save_checkpoint(model, optimizer, epoch, save_path, last_loss, best_loss):
    print("=> Saving checkpoint")
    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, os.path.join(save_path, "last.pt"))
    if last_loss < best_loss:
        best_loss = last_loss
        torch.save(checkpoint, os.path.join(save_path, "best.pt"))

    return best_loss

def load_check_point_to_use(checkpoint_file, model, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model"])

    return model

def load_checkpoint_to_continue(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file+'/last.pt', map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return model, epoch