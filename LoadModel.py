import torch
import torch.nn as nn
from torchtext.legacy import data

# Define TorchText Fields
TEXT = data.Field(sequential=True, tokenize='spacy', lower=True)
LABEL = data.LabelField()
csv_path = "OpenChat/data_ask.csv"
# Create a TorchText TabularDataset
fields = [('Sentence', TEXT), ('Label', LABEL)]
dataset = data.TabularDataset(
    path=csv_path,  # Replace with your CSV file path
    format='csv',
    fields=fields,
    skip_header=True
)

# Split dataset into train and test sets
train_data, test_data = dataset.split(split_ratio=0.8)  # 80% train, 20% test

# Build vocabulary
TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create iterators
BATCH_SIZE = 30000
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.Sentence),
    sort_within_batch=False,
    device=device
)

# Define the neural network model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))

# Initialize model
input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = len(LABEL.vocab)
print(input_dim, output_dim)
model = SimpleRNN(input_dim, embedding_dim, hidden_dim, output_dim).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.Sentence)
        loss = criterion(predictions, batch.Label)
        loss.backward()
        optimizer.step()

train(model, train_iterator, optimizer, criterion)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.Sentence)
            _, predicted = torch.max(predictions, 1)
            total_correct += (predicted == batch.Label).sum().item()
            total_count += len(batch.Label)
    return total_correct / total_count

accuracy = evaluate(model, test_iterator, criterion)
print(f"Accuracy: {accuracy * 100:.2f}%")

 # Save the trained model
torch.save(model.state_dict(), 'saved_model.pth')


def get_prediction(new_sentence):
    # Use the existing TEXT field for inference
    # new_sentence = "help me add to my shedule."

    # Preprocess the new sentence using the existing TEXT Field
    tokenized_sentence = TEXT.preprocess(new_sentence)
    indexed_sentence = [TEXT.vocab.stoi[token] for token in tokenized_sentence]
    tensor_sentence = torch.LongTensor(indexed_sentence).unsqueeze(1).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        output = model(tensor_sentence)

    # Get predicted class
    _, predicted = torch.max(output, 1)
    predicted_label = predicted.item()

    # Convert the predicted label back to the original class using TorchText's vocab
    predicted_class = LABEL.vocab.itos[predicted_label]

    return predicted_class

