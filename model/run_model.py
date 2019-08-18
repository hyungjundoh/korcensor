import word2vec.word2vec as w2v
import word2vec.csv_reader
from soynlp.tokenizer import RegexTokenizer
import model
import csv
import random
import numpy as np


class data_manager():
    def __init__(self, corpus_path, key_vector_path):
        self.corpus_path = corpus_path
        self.key_vector_path = key_vector_path

    # Prepare the corpus from given corpus path
    def prepare_corpus(self, ignore_list, model_length):
        tokenizer = RegexTokenizer()
        data_array = np.array()
        with open(self.corpus_path, newline='') as corpus_file:
            reader = csv.reader(corpus_file)
            for row in reader:
                sentence = row[0]
                label = row[1]

                if label == '1':
                    label = np.array([1, 0])
                else:
                    label = np.array([0, 1])

                tokenized_sentence = tokenizer.tokenize(str(sentence))
                clean_sentence = [
                    elem for elem in tokenized_sentence if csv_reader.is_valid_word(elem, ignore_list)]

                vector_list = [w2v.get_vector(elem) for elem in clean_sentence]
                np_data_array = np.asarray(vector_list)
                while (len(vector_list) < model_length):
                    np_data_array.append(np.zeros(100))

                data_array.append((np_data_array, label))

        train_input = [elem[0] for elem in data_array]
        train_label = [elem[1] for elem in data_array]

        return (train_input, train_label)
