file = open('royal_data.txt' , 'r')
# This line opens a file named royal_data.txt in read mode ('r').
# The open function returns a file object, which is assigned to the variable file.
royal_data = file.readlines()
# This line reads all the lines from the file and stores them in a list called royal_data.
# Each element in the list represents a line from the file, including the newline character at the end of each line.
file.close()
# It is important to close the file to free up system resources and to ensure that any changes made to the file are properly saved.
# print(royal_data)

# ['The future king is the prince\n', 'Daughter is the princess\n', 'Son is the prince\n', 'Only a man can be a king\n', 'Only a woman can be a queen\n', 'The princess will be a queen\n', 'The prince is a strong man\n', 'The princess is a beautiful woman\n', 'Prince is only a boy now\n', 'Prince will be king\n', 'A boy will be a man']
# lowercase  ,  removing \n char , and tokenize
for i in range(len(royal_data)):
    royal_data[i] = royal_data[i].lower().replace('\n' , ' ')
# print(royal_data)
#['the future king is the prince ', 'daughter is the princess ', 'son is the prince ', 'only a man can be a king ', 'only a woman can be a queen ', 'the princess will be a queen ', 'the prince is a strong man ', 'the princess is a beautiful woman ', 'prince is only a boy now ', 'prince will be king ', 'a boy will be a man']
# Removing stop words and tokenizing
stopwords = ['the' , 'is', 'only' ,'a','can' ,'be', 'will' ,'now' ]
filter_data = []
# for i in royal_data:
#     for word in i.split():
#         if word not in stopwords:
#             filter_data.append(word)
# print(filter_data)
#['future', 'king', 'prince', 'daughter', 'princess', 'son', 'prince', 'man', 'king', 'woman', 'queen', 'princess', 'queen', 'prince', 'strong', 'man', 'princess', 'beautiful', 'woman', 'prince', 'boy', 'prince', 'king', 'boy', 'man']
# we do need in this format so

for i in royal_data:
    temp = []
    for word in i.split():
        if word not in stopwords:
            temp.append(word)
    filter_data.append(temp)
print(filter_data)
# [['future', 'king', 'prince'], ['daughter', 'princess'], ['son', 'prince'], ['man', 'king'], ['woman', 'queen'], ['princess', 'queen'], ['prince', 'strong', 'man'], ['princess', 'beautiful', 'woman'], ['prince', 'boy'], ['prince', 'king'], ['boy', 'man']]


# creating Bigrams  :
bigrams = []
for word_list in filter_data:
    for i in range(len(word_list)-1):
        for j in range(i+1 ,len(word_list)):
            bigrams.append([word_list[i], word_list[j]])
            bigrams.append([word_list[j], word_list[i]]) #we need flipped also
print(bigrams)
#[['future', 'king'], ['king', 'future'], ['future', 'prince'], ['prince', 'future'], ['king', 'prince'], ['prince', 'king'], ['daughter', 'princess'], ['princess', 'daughter'], ['son', 'prince'], ['prince', 'son'], ['man', 'king'], ['king', 'man'], ['woman', 'queen'], ['queen', 'woman'], ['princess', 'queen'], ['queen', 'princess'], ['prince', 'strong'], ['strong', 'prince'], ['prince', 'man'], ['man', 'prince'], ['strong', 'man'], ['man', 'strong'], ['princess', 'beautiful'], ['beautiful', 'princess'], ['princess', 'woman'], ['woman', 'princess'], ['beautiful', 'woman'], ['woman', 'beautiful'], ['prince', 'boy'], ['boy', 'prince'], ['prince', 'king'], ['king', 'prince'], ['boy', 'man'], ['man', 'boy']]

all_words = [] #for unique words
for i in royal_data:
    for word in i.split():
        if word not in stopwords:
            all_words.append(word)
all_words = list(set(all_words))
all_words.sort()
print(all_words)
print(len(all_words))
# ['beautiful', 'boy', 'daughter', 'future', 'king', 'man', 'prince', 'princess', 'queen', 'son', 'strong', 'woman']
# 12

# ONE_HOT_ENCODING
# assigning all the uniqe words a token
word_dict ={}
counter =  0
for words in all_words:
    word_dict[counter] = words
    counter= counter+1
print(word_dict)
# {0: 'beautiful', 1: 'boy', 2: 'daughter', 3: 'future', 4: 'king', 5: 'man', 6: 'prince', 7: 'princess', 8: 'queen', 9: 'son', 10: 'strong', 11: 'woman'}

import numpy as np
onehot_data  = np.zeros((len(all_words), len(all_words)))
# print(onehot_data) 12 x 12 0 matrix
for i in range(len(all_words)):
    onehot_data[i][i] =1
# print(onehot_data)     Indentity matrix
onehot_dict = {}
for i in range(len(all_words)):
    onehot_dict[all_words[i]] = onehot_data[i]
print(onehot_dict)
#{'beautiful': array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'boy': array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'daughter': array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'future': array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]), 'king': array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), 'man': array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]), 'prince': array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), 'princess': array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), 'queen': array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]), 'son': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]), 'strong': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), 'woman': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])}
for word in onehot_dict:
    print(word , ":",onehot_dict[word])
# beautiful : [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# boy : [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

X = []
Y = []

for bi in bigrams:
    X.append(onehot_dict[bi[0]])
    Y.append(onehot_dict[bi[1]])
#print(X) [array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]), array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]), array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])]
#print(Y) [array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]), array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]), array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]), array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]), array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]


X = np.array(X)
Y= np.array(Y)


print(X)
print("bhai bawal ho gaya ye to")
print(Y)

# MODEL_TO_CREATE
#1. output and input same size
import tensorflow as tf
from tensorflow import keras
from keras import  Sequential
from keras.layers import Dense

vocab_size = len(onehot_data[0])
embedding_size =  2
model = Sequential([
        Dense(embedding_size, activation="linear"),
        Dense(vocab_size, activation="softmax")
])
model.compile(loss="categorical_crossentropy" , optimizer= "adam")
model.fit(X,Y ,epochs=1000)

# we have trained the model we just need to extract the weights whill will act as word embeddings
weights = model.get_weights()[0]

# print(weights) we just need the weight among hidden layer and output layer

# word_dict
# {0: 'beautiful', 1: 'boy', 2: 'daughter', 3: 'future', 4: 'king', 5: 'man', 6: 'prince', 7: 'princess', 8: 'queen', 9: 'son', 10: 'strong', 11: 'woman'}

# all_words
# ['beautiful', 'boy', 'daughter', 'future', 'king', 'man', 'prince', 'princess', 'queen', 'son', 'strong', 'woman']


# Inverting word_dict to get indices for words
index_dict = {v: k for k, v in word_dict.items()}

# Create the word embeddings dictionary
word_embeddings = {}
for word in all_words:
    word_index = index_dict[word]  # Get the index of the word
    word_embeddings[word] = weights[word_index]  # Get the embedding for the word

# Print the word embeddings dictionary
print(word_embeddings)

import matplotlib.pyplot as plt

# plt.figure(figsize = (10, 10))
for word in all_words:
    coord = word_embeddings[word]
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, (coord[0], coord[1]))

plt.savefig('img.jpg')


#
# import matplotlib.pyplot as plt
#
# # plt.figure(figsize = (10, 10))
# for word in list(word_dict.keys()):
#     coord = word_embeddings.get(word)
#     plt.scatter(coord[0], coord[1])
#     plt.annotate(word, (coord[0], coord[1]))

# plt.savefig('img.jpg')