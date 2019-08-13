from keras import layers, models


class model_seq(models.Model):
    def __init__(self, max_features, maxlen, dimension): # dimension: 출력 후 벡터 크기
        # RNN_LSTM
        x = layers.Input((maxlen,))
        h = layers.Embedding(max_features, dimension)(x)
        h = layers.LSTM(dimension, dropout=0.5, recurrent_dropout=0.5)(h)
        # ANN
        h = layers.Dense(dimension, activation="relu")(h)
        y = layers.Dense(dimension, activation="softmax")(h)

        super().__init__(x, y)

        self.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


if __name__ == "__main__":
    ## model_seq(20000,50,100)