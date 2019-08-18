from keras import layers, models
from keras.models import load_model, model_from_json
import run_model as model_manager
import numpy as np

ignore_list = ['+', '=', '<', '>', '(', ')', '\\', ':', '.', "'", '*', '-', '&', '1', '2', '3', '4', '5', '6', '7',
               '8', '9', '0', '!', '?', 'it', "=", ",", '.', ',', 'jpg', 'gif', '"', 'gis', 'JPG', 'n', 'ㄹ', 't']


class model(models.Model):
    def __init__(self, model_length=100, batch_size=100, inputs=None, outputs=None, name=None):
        self.model_length = model_length
        self.batch_size = batch_size
        if (inputs != None and outputs != None):
            super().__init__(inputs, outputs)

    # Build model with input shape 100, model_length (where 100 is dimension of w2v and model_length is length of LSTM sequence)

    def build_model(self):  # dimension: 출력 후 벡터 크기
        # RNN_LSTM
        x = layers.Input(
            batch_shape=(None, self.model_length, 100))
        h = layers.LSTM(self.model_length)(x)
        # ANN
        h = layers.Dense(self.model_length, activation="relu")(h)
        y = layers.Dense(2, activation="softmax")(h)

        super().__init__(x, y)

        self.compile(loss="categorical_crossentropy",
                     optimizer="adam", metrics=["accuracy"])

    # Train the model by calling model from the manager
    def train(self, corpus_path, model_path, key_vector_path, ignore_list, validation_split, epochs):
        manager = model_manager.data_manager(key_vector_path, model_path)
        corpus_input, corpus_label = manager.prepare_corpus(
            ignore_list, self.model_length, corpus_path)
        self.fit(np.array(corpus_input), np.array(corpus_label), self.batch_size,
                 epochs, validation_split=validation_split)

    # Save the model and its weights to given path
    def save_model(self, json_filename, weight_filename):
        with open(json_filename, 'w') as file:
            file.write(self.to_json())
        self.save_weights(weight_filename)
        print("Saved weights successfully")


# Used to load model from
class model_loader():
    def __init__(self, json_filename, weight_filename, corpus_path, key_vector_path, ignore_list, model_length):
        with open(json_filename, 'r') as file:
            loaded_from_json = file.read()
            loaded_model = model_from_json(
                loaded_from_json, custom_objects={'model': model})
            loaded_model.load_weights(
                weight_filename)
            self.model = loaded_model

        self.manager = model_manager.data_manager(key_vector_path, model_path)
        self.ignore_list = ignore_list
        self.model_length = model_length

    def predict(self, sentence):
        vector_array = self.manager.convert_to_vector_list(
            self.ignore_list, self.model_length, sentence)
        return self.model.predict(np.array([vector_array]), steps=1)


if __name__ == "__main__":
    json_filename = "model1.json"
    weight_filename = "weights1.h5"
    corpus_path = "../../dataset/dataset1.csv"
    model_path = "word2vec/word2vec.model"
    key_vector_path = "word2vec/word2vec.kv"

    model_length = 100
    batch_size = 100
    epochs = 3

    my_model = model(model_length, batch_size)
    my_model.build_model()
    my_model.train(corpus_path, model_path,
                   key_vector_path, ignore_list, 0.2, epochs)
    print("fit model")
    my_model.save_model(json_filename, weight_filename)
    loaded_model = model_loader(
        json_filename, weight_filename, corpus_path, key_vector_path, ignore_list, model_length)

    sentence = input()
    result = loaded_model.predict(sentence)
    print("Predicted result: ")
    print(result)
    if (result[0][1] > result[0][0]):
        print("Censor")
    else:
        print("UnCensor")
    # model_seq(20000,50,100)
