import word2vec.word2vec as w2v
import word2vec.csv_reader as csv_reader
from soynlp.tokenizer import RegexTokenizer
import model
import csv
import random
import numpy as np


class data_manager():
    def __init__(self, corpus_path, key_vector_path):
        self.corpus_path = corpus_path
        self.key_vector_path = key_vector_path

    def convert_to_vector_list(self, ignore_list, model_length, sentence):
        tokenizer = RegexTokenizer()
        tokenized_sentence = tokenizer.tokenize(str(sentence))
        myw2v = w2v.word2vec("word2vec/word2vec.model")
        myw2v.load_keyvector("word2vec/word2vec.kv")
        clean_sentence = [
            elem for elem in tokenized_sentence if csv_reader.is_valid_word(elem, ignore_list)]

        vector_list = [myw2v.get_vector(elem)for elem in clean_sentence]
        while (len(vector_list) < model_length):
            vector_list.append([0]*100)

        if (len(vector_list) > 100):
            vector_list = vector_list[:100]
        # Prepare the corpus from given corpus path

    def prepare_corpus(self, ignore_list, model_length):
        tokenizer = RegexTokenizer()
        data_list = []
        label_list = []
        myw2v = w2v.word2vec("word2vec/word2vec.model")
        myw2v.load_keyvector("word2vec/word2vec.kv")
        with open(self.corpus_path, newline='') as corpus_file:
            reader = csv.reader(corpus_file)
            for row in reader:
                sentence = row[0]
                label = row[1]

                if label == '1':
                    label = [1, 0]
                else:
                    label = [0, 1]

                tokenized_sentence = tokenizer.tokenize(str(sentence))
                clean_sentence = [
                    elem for elem in tokenized_sentence if csv_reader.is_valid_word(elem, ignore_list)]

                vector_list = [myw2v.get_vector(elem)
                               for elem in clean_sentence]
                while (len(vector_list) < model_length):
                    vector_list.append([0]*100)

                if (len(vector_list) > 100):
                    vector_list = vector_list[:100]
                # print(np.array(vector_list).shape)
                data_list.append(np.array(vector_list))
                label_list.append(np.array(label))

        train_input = data_list
        train_label = label_list

        return (train_input, train_label)
