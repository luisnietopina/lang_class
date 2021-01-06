import re
import json
from flask import Flask, jsonify, request
import torch
from torch import LongTensor
from bpemb import BPEmb
from src import LanguageClassifier


app = Flask(__name__)

MODEL_PATH = './models/model_15k-200h.pt'
CLASSES = {0:'Danish', 1:'Swedish', 2:'Norwegian'}


def init_model(bpe):
    '''
    Initializes a trained model for classification of sentences.
    '''

    VOCAB_SIZE = bpe.vocab_size
    EMBED_DIM = bpe.dim
    HIDDEN_SIZE = 200
    LSTM_LAYERS = 1
    NUM_CLASS = len(CLASSES)
    EPOCHS = 10

    model = LanguageClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, LSTM_LAYERS, NUM_CLASS, bpe)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    return model

def preproc(sent, bpe):
    '''
    Preprocess a single sentence to adapt it to the data format used to train the model.
    '''

    pattern = re.compile(r'[^\w ]+', re.UNICODE)
    sent = pattern.sub('', sent.lower())
    if sent == '':
        return None, None

    ids = bpe.encode_ids(sent)
    length = [len(ids)]

    return LongTensor(ids), LongTensor(length)

def classify_language(sent):
    '''
    Feeds a preprocessed sentence to the model and returns its class representing its language.
    '''

    bpe = BPEmb(lang='multi')
    model = init_model(bpe)

    ids, length = preproc(sent, bpe)
    if ids is None:
        return None
    ids = ids.view(ids.size()[0], 1)

    with torch.no_grad():
        lang_pred = model(ids, length)
        lang = CLASSES[lang_pred.argmax(1).item()]

    return lang

@app.route('/lang_class', methods=['POST'])
def lang_class():
    '''
    Listens for POST requests containing a sentence for classification with a trained model.
    Expects a request containing a string `sentence` in JSON format.
    '''

    if request.method == 'POST':
        print(request.data)
        data = request.get_json()
        sent = data['sentence']

        lang = classify_language(sent)

        return jsonify({'sentence': sent, 'language': lang})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
