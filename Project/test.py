from keras import layers, models
from keras.models import load_model, model_from_json
import numpy as np
import matplotlib.pyplot as plt
import pydot
import main

# Used to load model


if __name__ == '__main__':
    json_filename = "trained_model/model1.json"
    weight_filename = "trained_model/weights1.h5"
    corpus_path = "labeled_dataset/labeled.csv"
    model_path = "word2vec/w2v_model/word2vec.model"
    key_vector_path = "word2vec/w2v_model/word2vec.kv"
    train_graph_save_path = "graphs/train_graph.png"
    model_graph_save_path = "graphs/model_graph.png"

    model_length = 100

    loaded_model = main.model_loader(
        json_filename, weight_filename, corpus_path, key_vector_path, main.ignore_list, model_length)

    print("Please type in number of times you want to test")
    num_repeat = input()

    for i in range(0, int(num_repeat)):
        sentence = input()
        result = loaded_model.predict(sentence)

        print("Predicted result: ")
        print(result)
        if (result[0][1] > result[0][0]):
            print("Censor")
            print("With possibility of " + str(result[0][1] * 100) + "%")
        else:
            print("UnCensor")
            print("With possibility of " + str(result[0][0] * 100) + "%")
