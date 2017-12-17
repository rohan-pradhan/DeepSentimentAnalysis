'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as  pd




max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

tk = text.Tokenizer(num_words=max_features, lower=True, split=" ")
df = pd.read_csv('./bootstrap.csv', encoding = "ISO-8859-1")
# print (df.shape)
df = df[~df['sentiment'].isin(['Z','z'])]
# print (df.shape)
df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x=="P" else 0)
# print (df["sentiment"])


print('Loading data...')
(x_train, y_train), (b, g) = imdb.load_data(num_words=max_features)
x_test = df.text.astype(str).values.tolist()
# print (x_test)
y_test = df['sentiment'].values.tolist()
tk.fit_on_texts(x_test)
x_test = tk.texts_to_sequences(x_test)
# cleprint ("XTEST")
# print (x_test)
# print (x_train, y_train)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)