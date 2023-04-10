import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    model.to(device)
    test_img1 = transform(Image.open("data/train/image/000000000009.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    model.train()


def save_checkpoint(model, optimizer, epoch, save_path):
    print("=> Saving checkpoint")
    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, os.path.join(save_path, "last.pt"))
    # if np.mean(test_loss) < best_loss:
    #     best_loss = np.mean(test_loss)
    #     torch.save(checkpoint, os.path.join(save_path, "best.pt"))


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