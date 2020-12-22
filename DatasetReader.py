import pandas as pd
import spacy
import re
from itertools import chain


def read_text_file(text_file_path):
    try:
        tex_content = ""
        with open(text_file_path) as f:
            tex_content = f.read().rstrip("\n")
    except FileNotFoundError as file_not_found:
        print("File not found error ", file_not_found)
    except Exception as exception:
        print("Error while reading csv ", exception)

    return tex_content


def clean_sent(sent):
    sent = sent.lower()
    sent = str(sent).replace('(', ' ')
    sent = str(sent).replace(')', ' ')
    return sent


def found_entities(sentence, entities):
    tagged_entities = []
    clean_sentence = clean_sent(sentence)

    for entity in entities:
        entity = entity.lower()
        name_entity = [(entity_match.start(), entity_match.end(), 'jname') for entity_match in
                       list(re.finditer(entity, clean_sentence))]
        if len(name_entity) > 0:
            tagged_entities.append(name_entity)

    if len(tagged_entities) > 0:
        entities_index = list(chain.from_iterable(tagged_entities))
        sentence = str(sentence).replace('\n', ' ').replace('\t', ' ')
        sent_tuple = str((sentence, {"entities": entities_index}))
        with open('train_data.txt', 'a') as the_file:
            the_file.write(sent_tuple + "\n")


def entity_tag_spacy_format(text_sent, entities):
    sentence_count = 0
    for sentence in text_sent.sents:
        found_entities(sentence.text, entities)
        sentence_count = sentence_count + 1
    return sentence_count


class DatasetReader:

    def __init__(self):
        self.base_path_train = "input/train_data/"
        self.base_path_test = "input/test_data/"

        self.train_csv_filepath = "input/train.csv"
        self.test_csv_filepath = "input/test.csv"
        self.train_dir = "input/train_data/"
        self.test_dir = "input/test_data/"
        self.sp = spacy.load('en_core_web_sm')

    def read_csv(self, data_type="test"):
        try:
            metadata_csv = pd.DataFrame()
            if data_type == "test":
                metadata_csv = pd.read_csv(self.test_csv_filepath)
            else:
                metadata_csv = pd.read_csv(self.train_csv_filepath)
        except FileNotFoundError as exception:
            print("Error in reading csv format. Please check  file type", data_type, exception)
        except Exception as exception:
            print("Error while reading csv ", exception)
        return metadata_csv


if __name__ == '__main__':

    cls_reader = DatasetReader()
    df = cls_reader.read_csv("train")
    print("Total training data points ", df.shape)

    for index, row in df.iterrows():
        if index > 300:
            break
        filename = row['File_Name']
        tagged_names = row['Name'].split(',')

        filepath = cls_reader.base_path_train + "" + filename
        file_content = read_text_file(filepath)
        text_sentences = cls_reader.sp(file_content)
        sent_count = entity_tag_spacy_format(text_sentences, tagged_names)
        print(index, "Total Number of sentences are : ", sent_count, " for file ", filename)
