import random
import spacy
import ast
from pathlib import Path
from spacy.util import minibatch, compounding


class TrainAlgorithm:

    def __init__(self):
        self.spacy_train_data_filename = "train_data.txt"
        self.output_dir = "output/"

    def read_spacy_train_data(self):
        '''

        :return:  file_content =: Content of file
        '''
        file_content = []
        with open(self.spacy_train_data_filename) as f:
            for line in f:
                cl_line = ast.literal_eval(line)
                file_content.append(cl_line)
        return file_content

    def spacy_ner(self, TRAIN_DATA):
        '''

        :param TRAIN_DATA:
        :return:
        '''
        LABEL = ['jname']
        nlp = spacy.blank('en')
        print("Created blank 'en' model")
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        else:
            ner = nlp.get_pipe('ner')

        for i in LABEL:
            ner.add_label(i)

        optimizer = nlp.begin_training()
        n_iter = 100
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                batches = minibatch(TRAIN_DATA,
                                    size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    # Updating the weights
                    try:
                        nlp.update(texts, annotations, sgd=optimizer,
                                   drop=0.35, losses=losses)
                        print('Losses', losses)
                    except Exception as error:
                        print(error)
                        continue

        # Save model

        if self.output_dir is not None:
            output_dir = Path(self.output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.meta['name'] = "spacy_ner_legal"
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)


    # def spacy_ner(self, TRAIN_DATA):
    #
    #     model = None
    #     n_iter = 10
    #
    #     nlp = spacy.blank("en")
    #     if "ner" not in nlp.pipe_names:
    #         ner = nlp.create_pipe("ner")
    #         nlp.add_pipe(ner, last=True)
    #     ner.add_label('jname')
    #     optimizer = nlp.begin_training()
    #
    #     with nlp.disable_pipes():
    #         # training a new model
    #         if model is None:
    #             nlp.begin_training()
    #         for itn in range(n_iter):
    #             random.shuffle(TRAIN_DATA)
    #             losses = {}
    #             # batch up the examples using spaCy's minibatch
    #             batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    #             for batch in batches:
    #                 texts, annotations = zip(*batch)
    #                 nlp.update(
    #                     texts,
    #                     annotations,
    #                     drop=0.5,
    #                     losses=losses,
    #                 )
    #             print("Losses", losses)
    #
    #     output_dir = Path("/output")
    #     if not output_dir.exists():
    #         output_dir.mkdir()
    #     nlp.to_disk(output_dir)
    #     print("Saved model to", output_dir)


if __name__ == '__main__':
    train_spacy = TrainAlgorithm()
    train_data = train_spacy.read_spacy_train_data()
    count = 0
    print(train_data)
    train_spacy.spacy_ner(train_data)
