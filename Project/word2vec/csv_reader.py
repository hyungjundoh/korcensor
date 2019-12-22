import csv
import glob
import os
from soynlp.tokenizer import RegexTokenizer


class reader():
    def __init__(self, filepath):
        self.filepath = filepath

    # Reads csv file from filepath and returns clean_sentence_list
    #  of sentences
    def read_csv(self):
        with open(self.filepath, newline='') as word_file:
            clean_sentence_list = []
            reader = csv.reader(word_file)
            for row in reader:
                clean_sentence = []
                for elem in row:
                    if len(elem) > 0:
                        clean_sentence.append(elem)
                clean_sentence_list.append(clean_sentence)
        return clean_sentence_list


def is_valid_word(word, ignore_list):
    for elem in ignore_list:
        if elem in word:
            return False
    return True


# Tokenizes the sentnece into list and eliminates invalid words
# returns list of toeknized sentences
def clean_csv(dataset_file_dir, merged_file_save_path, ignore_list):
    sentence_list = []
    for filepath in os.listdir(dataset_file_dir):
        if filepath.endswith(".csv"):
            entire_path = os.path.join(dataset_file_dir, filepath)
            with open(entire_path, newline="") as word_file:
                csv_reader = csv.reader(word_file)
                for row in csv_reader:
                    sentence_list.append(row)

    tokenized_sentence_list = []
    tokenizer = RegexTokenizer()
    count = 0

    for sentence in sentence_list:
        tokenized_sentence = tokenizer.tokenize(str(sentence))
        clean_sentence = [
            elem for elem in tokenized_sentence if is_valid_word(elem, ignore_list)]
        tokenized_sentence_list.append(clean_sentence)
        # print(tokenized_sentence)
        count += 1

    file = open(merged_file_save_path, 'w', encoding='utf-8', newline='')
    writer = csv.writer(file)
    for sentence in tokenized_sentence_list:
        writer.writerow(sentence)
    file.close()
