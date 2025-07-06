## Load the dataset
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import SimpleRNN, Embedding, Dense
from keras.src.utils import pad_sequences
from tensorflow.keras.datasets import imdb
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


max_features=10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
# print(y_train[0])
# word_index = imdb.get_word_index()
# #word_index
# reverse_word_index = {value: key for key, value in word_index.items()}
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in X_train[0]])
# print(decoded_review)

max_len=500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# print(X_train)

## Train a simple RNN

model=Sequential()
model.add(Embedding(max_features,128,input_length=max_len))
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

model.build(input_shape=(None, max_len))
model.summary()

## Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history=model.fit(X_train,y_train,epochs=10,
                  batch_size=32,
                  validation_split=0.2,
                  verbose=1,
                  callbacks=early_stopping)


## Save the model
model.save('simple_rnn_imdb.h5')