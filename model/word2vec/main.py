import numpy as np
import csv_reader
import word2vec as w2v


ignore_list = ['+', '=', '<', '>', '(', ')', '\\', ':', '.', "'", '*', '-', '&', '1', '2', '3', '4', '5', '6', '7',
               '8', '9', '0', '!', '?', 'it', "=", ",", '.', ',', 'jpg', 'gif', '"', 'gis', 'JPG', 'n', 'ㄹ', 't']


if __name__ == '__main__':
    csv_filepath = "merged_file.csv"
    word2vec_model_path = "word2vec.model"
    keyvector_save_path = "word2vec.kv"

    print("Merging csv...")
    csv_reader.clean_csv("../../dataset", "merged_file.csv", ignore_list)
    reader = csv_reader.reader(csv_filepath)
    word_corpus = reader.read_csv()
    # print(word_corpus)
    print("Starting to train...")

    w2v_manager = w2v.word2vec(model_path=word2vec_model_path, mode=False)
    w2v_manager.train(word_corpus)
    w2v_manager.save_keyvector(keyvector_save_path)
    w2v_manager.load_keyvector(keyvector_save_path)
    print(w2v_manager.get_vector('양놈'))
