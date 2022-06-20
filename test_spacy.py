# --------------------------------------------
# LDD- AI Powered Enterprise Search
# __author__ : Dnyaneshwar Dewadkar
#            : 7:21 PM 21/12/20
#            : test_spacy.py
#
# --------------------------------------------
import spacy
import pandas as pd
from DatasetReader import DatasetReader

output_dir = 'output/'
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)


def predicted_entities(text_sentences):
    entities = []
    for sentence in text_sentences.sents:
        doc2 = nlp2(sentence.text)
        for ent in doc2.ents:
            if ent.label_ == 'jname':
                entities.append(ent.text)

    return entities


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


if __name__ == '__main__':
    filepath = 'input/test.csv'
    dsr = DatasetReader()
    df = dsr.read_csv('test')
    print("Total training data points ", df.shape)
    # df_pred = pd.DataFrame(columns=['File_Name', 'Name'])
    for index, row in df.iterrows():
        # if index > 3:
        #     break
        filename = row['File_Name']
        filepath = dsr.base_path_test + "" + filename
        file_content = read_text_file(filepath)
        text_sentences = dsr.sp(file_content)
        entities = predicted_entities(text_sentences)
        df.loc[index, 'Name'] = ','.join(entities)
        print(index, "processing file ", filename)

    print(df.head(10))
    df.to_csv('test_prediction.csv')
    print("NER Test Process Completed")
