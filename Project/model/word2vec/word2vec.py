import numpy as np
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
import word2vec.csv_reader
import os
import multiprocessing


class word2vec:
    # model_path : path of the model
    # mode : True if Cbow, False if Skip-gram
    def __init__(self, model_path, mode=True):
        self.model_path = model_path
        self.mode = mode

    # If trained model does not exist, train new model
    # If we already have trained model in model_path then call the model and keep on training
    def train(self, text_data):
        if (not os.path.exists(self.model_path)):
            print("Pretrained model does not exist... creating new model")
            model = Word2Vec(text_data, size=100, window=10,
                             min_count=5, workers=multiprocessing.cpu_count() - 1, iter=100, sg=int(self.mode))
            model.save(self.model_path)
        else:
            print("Loaded pretrained model from" + self.model_path)
            model = Word2Vec.load(self.model_path)
            model.train(
                text_data, total_examples=len(text_data), epochs=100)
            model.save(self.model_path)

    # Exports keyvector from trained model
    def save_keyvector(self, key_vector_path):
        if (not os.path.exists(self.model_path)):
            print("Could not find trained model.. please train the model first")
        else:
            model = Word2Vec.load(self.model_path)
            model.wv.save(key_vector_path)

    # Loads keyvector
    def load_keyvector(self, key_vector_path):
        if (not os.path.exists(key_vector_path)):
            print("Could not find key vector... please save key vector first")
        else:
            self.key_vector = KeyedVectors.load(key_vector_path, mmap='r')
            print("Successfully loaded key vector")

    # Gets embeded word vector from saved keyvector as numpy array
    def get_vector(self, word):
        try:
            array = self.key_vector[word]
            # print(len(array))
        except:
            array = [0]*100
            # print(array.shape)
        return array
