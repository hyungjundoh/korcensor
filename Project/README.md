
__Training Word2vec__ 

word2vec folder contains script files that trains word2vec model using gensim module

To execute, use

``` bash
python3 word2vec.py
```
Then, using csv files in Project/unlabeled_dataset folder, 100 dimension word2vec model will be generated

__Training LSTM model__

In project folder, you will see main.py and test.py and run_model.py

main.py file includes scripts to train the model
The corpus data will be loaded from csv files in Project/labeled_dataset folder

To train the model with prepared word2vec, use
```bash
python3 main.py
```

main.py will automatically save model to Project/trained_model folder.
.json file stores structure of the model and .h5 file will store trained weights.

It will also save its history figure in Project/graphs folder as train_graph.png

This directories can be changed by modifying variables in main.py

To test the trained model, use
```bash
python3 test.py
```


