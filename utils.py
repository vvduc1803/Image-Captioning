import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def read_caption(num_caption, vocab):
    str_caption = []
    for cap in num_caption[1:]:
        if vocab.itos[cap.item()] == "<EOS>":
            break
        str_caption.append(cap)

    return [vocab.itos[id.item()] for id in str_caption]

def print_examples(model, device, dataset, vocab, num_examples=5):
    model.eval()
    model.to(device)
    for example in range(num_examples):
        image, caption = dataset.__getitem__(example)
        image = image.to(device)
        print(f"Example {example+1} CORRECT: " + " ".join(read_caption(caption, vocab)))
        print(f"Example {example+1} OUTPUT: " + " ".join(model.caption_image(image.unsqueeze(0), vocab)))
        print('----------------------------------------------')
        fig, ax = plt.subplots()
        ax.imshow(dataset.img)
        ax.axis('off')
        fig.text(0.5, 0.05,
                 f"Example {example+1} CORRECT: " + " ".join(read_caption(caption, vocab)) + '\n' +
                 f"Example {example+1} OUTPUT: " + " ".join(model.caption_image(image.unsqueeze(0), vocab)),
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
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return model, epoch