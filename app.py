from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from keyword import *
import operator


vocab = np.load('mine_vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


print("+"*50)
print("vocabulary loaded")


embedding_size = 128
vocab_size = len(vocab)
max_len = 40


image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights.h5')

print("="*150)
print("MODEL LOADED")

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')


#resnet = load_model('resnet.h5')

print("="*150)
print("RESNET MODEL LOADED")




app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

    global model, resnet, vocab, inv_vocab

    img = request.files['file1']

    img.save('static/file.jpg')

    print("="*50)
    print("IMAGE SAVED")


    
    image = cv2.imread(r'static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))

    
    
    incept = resnet.predict(image).reshape(1,2048)

    print("="*50)
    print("Predict Features")


    text_in = ['startofseq']

    final = ''

    print("="*50)
    print("GETING Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)

        # Split text into sentences
        #sentenceList = split_sentences(final)

        #stoppath = "FoxStoplist.txt" #Fox stoplist contains "numbers", so it will not find "natural numbers" like in Table 1.1

        #stoppath = "SmartStoplist.txt"  #SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
        #stopwordpattern = build_stop_word_regex(stoppath)

        # generate candidate keywords

        #phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)

        # calculate individual word scores

        #wordscores = calculate_word_scores(phraseList)

        # generate candidate keyword scores

        #keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)

        #sortedKeywords = sorted(keywordcandidates.items(), key=operator.itemgetter(1), reverse=True)

        #totalKeywords = len(sortedKeywords)

        #print sortedKeywords[0:(totalKeywords / 3)]

        #rake = Rake("SmartStoplist.txt")
        #keywords = rake.run(final)
        #print (keywords)

    return render_template('after.html', data=final)

if __name__ == "__main__":
    app.run(debug=True)


