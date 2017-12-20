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
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM, Bidirectional
from keras.datasets import imdb
import glob
import pandas as  pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

stopWords = stopwords.words("english")
stemmer = PorterStemmer()

def format_sentence(sent):
	words = [word.lower() for word in word_tokenize(sent)]
	filtered_words = [stemmer.stem(word) for word in words if word not in stopWords]
	return " ".join(filtered_words)

paths = glob.glob("./aclImdb/train/pos/*.txt")
pos_reviews = []
pos_reviews_raw = []


for path in paths:
    reviewText = open(path, 'r', encoding="latin-1").read()
    pos_reviews_raw.append(format_sentence(reviewText))
    #pos_reviews_raw.append(reviewText)
    #f.value += 1

paths = glob.glob("./aclImdb/train/neg/*.txt")
neg_reviews = []
neg_reviews_raw = []

for path in paths:
    reviewText = open(path, 'r', encoding="utf8").read()
    neg_reviews_raw.append(format_sentence(reviewText))
    # neg_reviews_raw.append(reviewText)
    #f.value += 1




max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

# for sent in pos_reviews_raw:
# 	pos_reviews.append(format_sentence(sent))

# for sent in neg_reviews_raw:
# 	neg_reviews.append(format_sentence(sent))



tk = text.Tokenizer(num_words=max_features, lower=True, split=" ")
tk.fit_on_texts(pos_reviews_raw+neg_reviews_raw)
pos_reviews = tk.texts_to_sequences(pos_reviews_raw)
neg_reviews = tk.texts_to_sequences(neg_reviews_raw)
pos_review_sent = [1]*len(pos_reviews)
neg_review_sent = [0]*len(neg_reviews)

X = pos_reviews + neg_reviews
Y = pos_review_sent + neg_review_sent
x_train, x_val, y_train, y_val =train_test_split(X, Y, test_size=0.33, random_state=42)
print (x_val, y_val)
print (len(x_val))
print (len(y_val))
#x_train = X
#y_train = Y
#
#columns = ["text", "sentiment"]
#df_pos = pd.DataFrame(pos_reviews, columns=['text'])
#df_pos['sentiment'] = 1

print ("Printing df pos: ")
#print (df_pos)

df = pd.read_csv('./bootstrap.csv', encoding = "ISO-8859-1")
# print (df.shape)
df = df[~df['sentiment'].isin(['Z','z'])]
# print (df.shape)
df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x=="P" else 0)
# print (df["sentiment"])


print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_test = [format_sentence(i) for i in df.text.astype(str).values.tolist()]
# # print (x_test)
y_test = df['sentiment'].values.tolist()
#tk.fit_on_texts(x_test)
x_test = tk.texts_to_sequences(x_test)
# cleprint ("XTEST")
# print (x_test)
# print (x_train, y_train)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_val, y_val))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)