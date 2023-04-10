# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import os  # when loading file paths
import spacy  # for tokenizer
import torch
import config
import json

from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

# Download with: python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        # Initialize 2 dictionary: index to string and string to index
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        # Threshold for add word to dictionary
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class CoCoDataset(Dataset):
    def __init__(self, root_dir, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        captions_path = open(os.path.join(self.root_dir, config.captions), 'r')
        captions_file = json.load(captions_path)
        self.transform = transform

        # Get img, caption columns
        self.imageID_list = [captions['image_id'] for captions in captions_file['annotations']]
        self.captions_list = [captions['caption'] for captions in captions_file['annotations']]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions_list)
        print(self.vocab)

    def __len__(self):
        return len(self.imageID_list)

    def __getitem__(self, index):
        caption = self.captions_list[index]
        img_id = str((self.imageID_list[index])).zfill(12) + '.jpg'
        img = Image.open(os.path.join(self.root_dir, config.images, img_id)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
    root_folder,
    transform,
    batch_size=2,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = CoCoDataset(root_folder, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        'data/train', transform=transform
    )
    from utils import print_examples
    from model import ImgCaption_Model
    # model = ImgCaption_Model(256, 256, len(dataset.vocab), 1)
    # print_examples(model, 'cuda',dataset)
    # imgs, captions = dataset.__getitem__(1)
    # print(imgs.shape)
    # print(captions)
    # print(captions.shape)
    for x, y in loader:
        a = [[1], [2], [3]]
        print(a[:-1])
        print(y[:-1])
        print(y)
        break