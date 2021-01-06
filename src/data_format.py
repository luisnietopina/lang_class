from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from bpemb import BPEmb


class SentenceDataset(Dataset):
    '''
    Dataset container class to read and store training/validation data intended 
    to feed the classifier model through a DataLoader.
    The expected text file line format is

    <label (int 0 to 2)>\t<sentence>

    '''

    def __init__(self, filename, bpe=None):
        self.data = []
        if bpe is None:
            self.bpe = BPEmb(lang='multi')
        else:
            self.bpe = bpe
        self.read_data(filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read_data(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                try:
                    label, sent = line.strip().split('\t')
                except ValueError:
                    continue

                ids = self.bpe.encode_ids(sent)
                label = int(label)

                self.data.append((label, LongTensor(ids)))

def collate_fn(batch):
    '''
    Function to be passed to a DataLoader to process training batches from a SentenceDataset object.
    '''
    sentences = []
    labels = []
    lengths = []

    for label, sent in batch:
        sentences.append(sent)
        lengths.append(len(sent))
        labels.append(label)

    labels = LongTensor(labels)
    lengths = LongTensor(lengths)
    sentences = pad_sequence(sentences)

    return sentences, lengths, labels

