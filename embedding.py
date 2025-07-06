### sentences
from keras.src.utils import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

voc_size=10000
sent_length = 8
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

print(sent)

one_hot_rep = [one_hot(words, voc_size) for words in sent]
# print(one_hot_rep)

embedding_docs = pad_sequences(one_hot_rep, padding='post', maxlen=sent_length)
print(embedding_docs)

dim = 10
model=Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_length))
model.compile('adam','mse')
model.build(input_shape=(None, sent_length))

model.summary()

result = model.predict(embedding_docs[0:1])
print(result)