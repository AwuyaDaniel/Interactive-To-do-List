import os

import torch
import pandas as pd
import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

spacy_eng = spacy.load('en_core_web_sm')


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
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
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]


class ToneDataSet(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)  # Fix: Use the provided caption_file parameter
        self.transform = transform

        self.sentence = self.df['Sentence']
        self.label = self.df['Label']
        # build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.label.tolist())

    def __len__(self):
        return len(self.df)

    # def __getitem__(self, index):
    #     label = self.label[index]
    #     sentence = self.sentence[index]
    #
    #     numerical_sentence = [self.vocab.stoi["<SOS>"]]
    #     numerical_sentence += self.vocab.numericalize(sentence)
    #     numerical_sentence.append(self.vocab.stoi["<EOS>"])  # Fix: Remove extra list around EOS
    #
    #     numerical_label = self.vocab.numericalize(label)
    #
    #     return numerical_label, torch.tensor(numerical_sentence)
    def __getitem__(self, index):
        label = self.label[index]
        sentence = self.sentence[index]

        # For simplicity, consider the first label as the target label
        label = label.split(',')[0]

        numerical_sentence = [self.vocab.stoi["<SOS>"]]
        numerical_sentence += self.vocab.numericalize(sentence)
        numerical_sentence.append(self.vocab.stoi["<EOS>"])

        numerical_label = self.vocab.numericalize(label)

        # Ensure the index is within the range of expected classes
        numerical_label = [idx % 30 for idx in numerical_label]

        return numerical_label[0], torch.tensor(numerical_sentence)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        pointer = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = pad_sequence(target, batch_first=False, padding_value=self.pad_idx)
        return pointer, target


def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True
):
    dataset = ToneDataSet(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader


# def main():
#     dataloader = get_loader("OpenChat/labeled_data.csv", annotation_file='OpenChat/labeled_data.csv', transform=None)
#     for (idx, caption) in enumerate(dataloader):
#         print(caption[1].shape)  # Fix: Correct the print statement


# Define a simple example model
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        prediction = self.fc(hidden.squeeze(0))
        return prediction


def train(model, train_loader, criterion, optimizer, num_epochs=5):
    # for epoch in range(num_epochs):
    #     for labels, sentences in train_loader:
    #         optimizer.zero_grad()
    #         output = model(sentences)
    #
    #         # Convert labels to tensor
    #         labels_tensor = torch.tensor(labels)
    #
    #         loss = criterion(output, labels_tensor)
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    # def train(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        for labels, sentences in train_loader:
            optimizer.zero_grad()
            output = model(sentences)

            # Convert labels to tensor
            labels_tensor = torch.tensor(labels)

            # Ensure the indices are within the range of expected classes
            labels_tensor = labels_tensor % 30

            loss = criterion(output, labels_tensor)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


def save_model(model, optimizer, filename='model.pth'):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, filename)
    print(f'Model saved as {filename}')


def load_model(model, optimizer, filename='model.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Model loaded from {filename}')


def save_model(model, optimizer, save_dir='models', filename='model.pth'):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, filepath)
    print(f'Model saved as {filepath}')


def load_model(model, optimizer, load_dir='models', filename='model.pth'):
    filepath = os.path.join(load_dir, filename)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Model loaded from {filepath}')


def main():
    # Initialize and train the model
    dataset = ToneDataSet(root_dir="OpenChat/labeled_data.csv", caption_file='OpenChat/labeled_data.csv',
                          transform=None)
    model = SimpleModel(vocab_size=len(dataset.vocab), embedding_dim=100, hidden_dim=128, output_dim=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_loader("OpenChat/labeled_data.csv", annotation_file='OpenChat/labeled_data.csv', transform=None)

    train(model, train_loader, criterion, optimizer)

    # Save the trained model to the 'saved_models' directory
    save_model(model, optimizer, save_dir='saved_models')

    # Load the saved model from the 'saved_models' directory
    loaded_model = SimpleModel(vocab_size=len(dataset.vocab), embedding_dim=100, hidden_dim=128, output_dim=4)
    loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.001)
    load_model(loaded_model, loaded_optimizer, load_dir='saved_models')

    # Now you can use loaded_model for predictions


if __name__ == "__main__":
    main()
