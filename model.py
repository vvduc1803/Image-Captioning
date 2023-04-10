import torch
import torch.nn as nn
import torchvision.models as models

from torchinfo import summary

class CNN(nn.Module):
    def __init__(self, embed_size=256, train_model=False):
        super().__init__()

        self.model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights)
        if not train_model:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier.requires_grad_(True)
        self.model.classifier = nn.Sequential(nn.Linear(1408, embed_size),
                                              nn.ReLU(),
                                              nn.Dropout(0.5))

    def forward(self, x):
        return self.model(x)

class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers, embed_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.drop_out(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden)

        return outputs

class ImgCaption_Model(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.CNN = CNN(embed_size)
        self.RNN = RNN(hidden_size, vocab_size, num_layers, embed_size)

    def forward(self, images, captions):

        features = self.CNN(images)
        outputs = self.RNN(features, captions)

        return outputs

    def caption_image(self, image, vocab, max_length=50):
        result = []

        with torch.inference_mode():
            features = self.CNN(image)
            state = None
            for _ in range(max_length):
                hidden, state = self.RNN.lstm(features, state)
                output = self.RNN.linear(hidden)
                predict = output.argmax(axis=1)
                result.append(predict.item())
                features = self.RNN.embed(predict)

                if vocab.itos[predict.item()] == "<EOS>":
                    break

        return [vocab.itos[idx] for idx in result]

if __name__ == '__main__':
    pass