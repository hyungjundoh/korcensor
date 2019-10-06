import word2vec.word2vec as w2v
import word2vec.csv_reader as csv_reader
from soynlp.tokenizer import RegexTokenizer
from gensim.models import Word2Vec, KeyedVectors
import csv
import random
import numpy as np


class data_manager():
    def __init__(self, key_vector_path, model_path):
        self.key_vector_path = key_vector_path
        self.model_path = model_path

    def convert_to_vector_list(self, ignore_list, model_length, sentence):
        tokenizer = RegexTokenizer()
        tokenized_sentence = tokenizer.tokenize(str(sentence))
        print(self.key_vector_path)
        kv = KeyedVectors.load(self.key_vector_path, mmap='r')
        clean_sentence = [
            elem for elem in tokenized_sentence if csv_reader.is_valid_word(elem, ignore_list)]

        vector = []
        for elem in clean_sentence:
            try:
                array = kv[elem]
            except:
                array = [1] * 100
            vector.append(array)

        vector_list = []
        while (len(vector_list) < model_length):
            vector_list += vector

        if (len(vector_list) > model_length):
            vector_list = vector_list[:model_length]
        return np.array(vector_list)
        # Prepare the corpus from given corpus path

    def prepare_corpus(self, ignore_list, model_length, corpus_path):
        tokenizer = RegexTokenizer()
        data_list = []
        label_list = []
        myw2v = w2v.word2vec(self.model_path)
        myw2v.load_keyvector(self.key_vector_path)
        with open(corpus_path, newline='') as corpus_file:
            reader = csv.reader(corpus_file)
            for row in reader:
                sentence = row[0]
                label = row[1]

                # uncensored data
                if label == '1':
                    label = [1, 0]
                # Censored data
                else:
                    label = [0, 1]

                tokenized_sentence = tokenizer.tokenize(str(sentence))
                clean_sentence = [
                    elem for elem in tokenized_sentence if csv_reader.is_valid_word(elem, ignore_list)]

                vector = [myw2v.get_vector(elem)
                          for elem in clean_sentence]
                print("length: " + str(len(vector)))

                if(len(vector) > 0):
                    vector_list = []
                    while (len(vector_list) < model_length):
                        vector_list += vector

                    if (len(vector_list) > model_length):
                        vector_list = vector_list[:model_length]
                    # print(np.array(vector_list).shape)
                    data_list.append(np.array(vector_list))
                    label_list.append(np.array(label))

        train_input = data_list
        train_label = label_list

        return (train_input, train_label)
