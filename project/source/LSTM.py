import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras import models
from keras import optimizers
import nltk
from nltk.corpus import stopwords


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype = 'float32')

def remove_stopwords(old_dict):
    for key in stopwords.words():
        if key in old_dict.keys():
            del old_dict[key]
    return old_dict

class LSTM:

    def __init__(self, MAX_SEQUENCE_LENGTH, max_features, embed_size, embedding_matrix):
        self.model = models.Sequential()
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.max_features = max_features
        self.embed_size = embed_size
        self.embedding_matrix = embedding_matrix

    def set_up(self):
        self.model = models.Sequential()

        self.model.add(layers.Input(shape = (MAX_SEQUENCE_LENGTH,)))
        self.model.add(layers.Embedding(max_features, embed_size, weights=[embedding_matrix]))
        self.model.add(layers.Bidirectional(layers.LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(layers.GlobalMaxPool1D())
        self.model.add(layers.Dense(50, activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(6, activation="sigmoid"))

        self.model.summary()

        opt = optimizers.Adam(learning_rate=0.0005)
        self.model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = ["accuracy"])


if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_labels = train[labels].values
    examples = pd.DataFrame(["I love you", "You suck", "You are idiot"])
    examples.columns = ["comment_text"]
    # Parameters
    embed_size = 100
    max_features = 20000
    MAX_SEQUENCE_LENGTH = 100
    EPOCH = 5
    BATCH_SIZE = 32

    tokenizer = Tokenizer(num_words = max_features)
    tokenizer.fit_on_texts(list(train['comment_text']))
    train_input = pad_sequences(tokenizer.texts_to_sequences(train['comment_text']), maxlen = MAX_SEQUENCE_LENGTH)
    test_input = pad_sequences(tokenizer.texts_to_sequences(test['comment_text']), maxlen = MAX_SEQUENCE_LENGTH)
    examples_input = pad_sequences(tokenizer.texts_to_sequences(examples['comment_text']), maxlen = MAX_SEQUENCE_LENGTH)

    nltk.download('stopwords')
    # Reference: https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    EMBEDDING_FILE = '../data/glove.6B.100d.txt'
    
    embedding_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE,encoding="utf8"))

    all_embs = np.stack(embedding_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    word_index_without_sw = remove_stopwords(word_index)
    num_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embed_size))
    i = 0
    for word in word_index_without_sw.keys():
        if i >= num_words: 
            break
        if word in embedding_index.keys():
            embedding_matrix[i] = embedding_index[word]
        i += 1

    my_model = LSTM(MAX_SEQUENCE_LENGTH, max_features, embed_size, embedding_matrix)
    my_model.set_up()
    my_model.model.fit(train_input, train_labels, batch_size = BATCH_SIZE, epochs = EPOCH, validation_split = 0.2)
    predict_on_test = my_model.model.predict(test_input)
    out = pd.concat([test["id"], pd.DataFrame(predict_on_test)], axis=1, ignore_index = True)
    output_columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    out.columns = output_columns
    submission_file = open("../output/submission_LSTM.csv", "w")
    out.to_csv('../output/submission_LSTM.csv', index = False)
    submission_file.close()

