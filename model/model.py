from keras import layers, models
from keras.models import load_model, model_from_json
from run_model import data_manager

ignore_list = ['+', '=', '<', '>', '(', ')', '\\', ':', '.', "'", '*', '-', '&', '1', '2', '3', '4', '5', '6', '7',
               '8', '9', '0', '!', '?', 'it', "=", ",", '.', ',', 'jpg', 'gif', '"', 'gis', 'JPG', 'n', 'ㄹ', 't']


class model(models.Model):
    def __init__(self, model_length, batch_size):
        self.model_length = model_length
        self.batch_size = batch_size

    # Build model with input shape 100, model_length (where 100 is dimension of w2v and model_length is length of LSTM sequence)
    def build_model(self):  # dimension: 출력 후 벡터 크기
        # RNN_LSTM
        x = layers.Input(shape=(100, self.model_length),
                         batch_size=self.batch_size)
        h = layers.LSTM(self.model_length)(x)
        # ANN
        h = layers.Dense(self.model_length, activation="relu")(h)
        y = layers.Dense(2, activation="softmax")(h)

        super().__init__(x, y)

        self.compile(loss="categorical_crossentropy",
                     optimizer="adam", metrics=["accuracy"])

    # Train the model by calling model from the manager
    def train(self, corpus_path, key_vector_path, ignore_list, validation_split, epochs):
        manager = data_manager(corpus_path, key_vector_path)
        corpus_input, corpus_label = manager.prepare_corpus(
            ignore_list, self.model_length)
        self.fit(corpus_input, corpus_label, self.batch_size,
                 epochs, validation_split=validation_split)

    # Save the model and its weights to given path
    def save_model(self, json_filename, weight_filename):
        with open(json_filename, 'w') as file:
            file.write(self.to_json())
        self.save_weights(weight_filename)
        print("Saved weights successfully")

    # Load the model from given path
    def load_model(self, json_filename, weight_filename):
        with open(json_filename, 'r') as file:
            loaded_model = model_from_json(file)
            loaded_model.load_weights(weight_filename)
            self.model = loaded_model

    # Returns prediction from the
    def predict(self, input):
        if (hasattr(self, 'model')):
            return self.model.predict(input)
        else:
            return self.predict(input)


if __name__ == "__main__":
    my_model = model(100, 100)
    # model_seq(20000,50,100)
