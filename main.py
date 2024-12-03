import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

#https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt

filepath = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

#open file and extract/decode the text and make it all in lowercase to improve accuracy of the NN
text = open(filepath, "rb").read().decode(encoding='utf-8').lower()

#convert the text into numbers for NN

text = text[300000:800000]

characters = sorted(set(text))

#convert to numerical 

char_to_num = dict((c,n) for n, c in enumerate(characters))

num_to_char = dict((n,c) for n, c in enumerate(characters))

SEQ_LENGTH = 50
STEP_SIZE = 3

# sentences = []

# next_char = []

# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i: i+ SEQ_LENGTH])
#     next_char.append(text[i+SEQ_LENGTH])


# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool) # x[4,6,3] if there is a character there it will be true otherwise false
# y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

# for i, sen in enumerate(sentences):
#     for j, char in enumerate(sen):
#         x[i,j, char_to_num[char]] = 1

#     y[i,char_to_num[next_char[i]]]=1

# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQ_LENGTH,len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# model.fit(x,y, batch_size=256, epochs=4)

# model.save('TextGenAI.keras')

model = tf.keras.models.load_model('TextGenAI.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temp):
    start_index = random.randint(0, (len(text) - SEQ_LENGTH - 1))
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_num[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temp)
        next_character = num_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated


print("==========================RUN 1 LEN 300, TEMP 0.4 ==========================\n\n")
print(generate_text(300, 0.4))
print("==========================LEN 300, TEMP 0.6 ========================== \n\n")
print(generate_text(300, 0.6))
print("==========================RUN 1 LEN 300, TEMP 0.8 ==========================\n\n")
print(generate_text(300, 0.8))