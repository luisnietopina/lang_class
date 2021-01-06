import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from src import SentenceDataset, collate_fn
from src import LanguageClassifier


TRAINING_PATH = './data/training_data_15k.txt'
VAL_PATH = './data/validation_data_6k.txt'
TRAINED_MODEL_PATH = 'models/trained_lang_classifier.pt'

CLASSES = {0:'Danish', 1:'Swedish', 2:'Norwegian'}

BATCH_SIZE = 10
HIDDEN_SIZE = 200
LSTM_LAYERS = 1
NUM_CLASS = len(CLASSES)
EPOCHS = 10


def load_dataset(path, bpe=None):
    '''
    Data reading and formatting using class SentenceDataset; DataLoader serves batches as input to
    the model.
    '''

    dataset = SentenceDataset(path, bpe)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    return dataset, loader

def train(data):
    '''
    Performs a training epoch looping through data batches. Returns performance on training data 
    in terms of loss and accuracy.
    '''

    loss_total = 0
    acc_total = 0
    size_total = 0
    for x_batch, x_batch_lengths, y_batch in data:
        model.zero_grad()

        y_pred = model(x_batch, x_batch_lengths)

        loss = loss_function(y_pred, y_batch)
        loss_total += loss.item()
        loss.backward()
        optimizer.step()

        acc_total += (y_pred.argmax(1) == y_batch).sum().item()
        size_total += x_batch.size(1)
    scheduler.step()
    return loss_total / size_total, acc_total / size_total

def test(data):
    '''
    Performs classification of data batches for the purpose of validation during training.
    Returns loss and accuracy averaged over all batches.
    '''

    loss_total = 0
    acc_total = 0
    size_total = 0
    for x_batch, x_batch_lengths, y_batch in data:
        with torch.no_grad():
            y_pred = model(x_batch, x_batch_lengths)
            loss = loss_function(y_pred, y_batch)
            loss_total += loss.item()
            acc_total += (y_pred.argmax(1) == y_batch).sum().item()
            size_total += x_batch.size(1)
    return loss_total / size_total, acc_total / size_total


if __name__ == '__main__':

    print('Reading data...')

    training_set, training_ldr = load_dataset(TRAINING_PATH)
    _, val_ldr = load_dataset(VAL_PATH, training_set.bpe)

    print('Data loaded.')

    vocab_size = training_set.bpe.vocab_size
    embed_dim = training_set.bpe.dim

    model = LanguageClassifier(vocab_size, embed_dim, HIDDEN_SIZE, LSTM_LAYERS, NUM_CLASS, training_set.bpe)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)

    print('Training...')

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(training_ldr)
        val_loss, val_acc = test(val_ldr)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), "in %d minutes, %d secons" % (mins, secs))
        print(f'\tTraining loss:\t\t{train_loss:.4f}\tTraining acc:\t{train_acc*100:.1f}%')
        print(f'\tValidation loss:\t{val_loss:.4f}\tValidation acc:\t{val_acc*100:.1f}%')

    torch.save(model.state_dict(), TRAINED_MODEL_PATH)
    print('Finished training. Saved model in', TRAINED_MODEL_PATH)
