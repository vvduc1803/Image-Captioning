# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import os
import spacy  # for tokenizer
import torch
import config
import json

from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# Download with: python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold=5):
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

    def read_vocab(self, file_name='vocab.json'):
        """
        Load created vocabulary file and replace the 'index to string' and 'string to index' dictionary
        """
        vocab_path = open(file_name, 'r')
        vocab = json.load(vocab_path)
        new_itos = {int(key): value for key, value in vocab['itos'].items()}

        self.itos = new_itos
        self.stoi = vocab['stoi']

    def create_vocab(self, file_name='vocab.json'):
        # create json object from dictionary
        vocab = json.dumps({'itos': self.itos,
                            'stoi': self.stoi})

        # open file for writing, "w"
        f = open(file_name, "w")

        # write json object to file
        f.write(vocab)

        # close file
        f.close()

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class CoCoDataset(Dataset):
    def __init__(self, root_dir, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.freq_threshold = freq_threshold
        captions_path = open(os.path.join(self.root_dir, config.captions), 'r')
        captions_file = json.load(captions_path)
        self.transform = transform

        # Get img, caption columns
        self.imageID_list = [captions['image_id'] for captions in captions_file['annotations']]
        self.captions_list = [captions['caption'] for captions in captions_file['annotations']]

        # # Initialize vocabulary and build vocab
        # if not self.set_vocab:
        #     self.vocab = Vocabulary(self.freq_threshold)
        #     self.vocab.build_vocabulary(self.captions_list)
        #     self.vocab.create_vocab()
        # else:
        #     self.vocab = self.set_vocab

        # Load vocab file
        self.vocab = Vocabulary(self.freq_threshold)
        self.vocab.read_vocab()

    def __len__(self):
        return len(self.imageID_list)

    def __getitem__(self, index):

        # Load index caption and image
        caption = self.captions_list[index]
        img_id = str((self.imageID_list[index])).zfill(12) + '.jpg'
        self.img = Image.open(os.path.join(self.root_dir, config.images, img_id)).convert("RGB")

        # Transform image
        if self.transform:
            img = self.transform(self.img)

        # Numericalized captions
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
    batch_size=16,
    num_workers=4,
    shuffle=True,
    pin_memory=True
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
    return dataset, loader



if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    train_dataset, train_loader = get_loader(root_folder=config.train,
                                                    transform=config.transform,
                                                    batch_size=config.batch_size,
                                                    num_workers=config.num_workers,
                                                    shuffle=True)
    from utils import plot_examples
    from model import ImgCaption_Model
    model = ImgCaption_Model(256, 256, len(train_dataset.vocab), 1)
    plot_examples(model, 'cuda', train_dataset, train_dataset.vocab)
    # imgs, captions = dataset.__getitem__(1)
    # print(imgs.shape)
    # print(captions)
    # print(captions.shape)
    # for x, y in loader:
    #     a = [[1], [2], [3]]
    #     print(a[:-1])
    #     print(y[:-1])
    #     print(y)
    #     break