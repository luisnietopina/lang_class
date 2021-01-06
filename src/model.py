from torch import tensor
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

class LanguageClassifier(nn.Module):
    '''
    Main class implementing the language classifier as a neural network:
    1. Embedding layer
    2. LSTM layer
    3. Fully connected layer with 3-unit output
    - Log-softmax output function
    '''

    def __init__(self, vocab_size, embed_dim, hidden_size, lstm_layers, num_class, bpe):
        super().__init__()

        # Hyperparams
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_class = num_class
        self.bpe = bpe

        # Layer definition
        self.embedding = nn.Embedding.from_pretrained(tensor(self.bpe.vectors))
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=self.lstm_layers)
        self.fc1 = nn.Linear(self.hidden_size, self.num_class)

        # FC layer weight initialization
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        if self.training:
            packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
            _, (hs, _) = self.lstm(packed)
        else:
            _, (hs, _) = self.lstm(embedded)
        hs = hs.view(hs.size()[1], hs.size()[2])
        out1 = self.fc1(hs)
        scores = F.log_softmax(out1, dim=1)

        return scores

