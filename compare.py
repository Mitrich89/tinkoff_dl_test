import argparse
import numpy as np
import pathlib
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from normalizer import AstTransformer


class Compare:

    def __init__(self, input_path):
        self.input_path = input_path
        self.files_names = []
        self.folders_paths = []
        self.answers = []

    def get_paths_and_filenames(self, input_path):
        with open(input_path, 'r') as f:
            paths = f.read()

        objects = paths.split()
        for path in objects:
            folder = []
            filename = path.split('/')[-1]
            self.files_names.append(filename)
            folders = path.split('/')
            for i in range(len(path.split('/')) - 1):
                folder.append(folders[i] + '/')
            folder = ''.join(folder)
            self.folders_paths.append(folder)

    def normalize_code(self, file_name, path1, path2):
        transformer = AstTransformer()
        transformed_origin = transformer.transform_file(pathlib.Path(f'{path1}{file_name}'))
        transformed_plagiat_1 = transformer.transform_file(pathlib.Path(f'{path2}{file_name}'))
        return transformed_origin, transformed_plagiat_1

    def compute_cos_sim(self, origin, plagiat) -> float:
        pair = [origin, plagiat]
        vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_vectors = vectorizer.fit_transform(pair)
        return cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

    def compute_avg_len(self, code_text: str) -> float:
        words = code_text.split()
        sum_of_lengths = sum(len(i) for i in words)
        return sum_of_lengths / len(words)

    def make_df(self, code_text: str, filename: str, cos_sim: float, target: int) -> pd.DataFrame:
        df = pd.DataFrame({'functions': code_text.count(' def ') + code_text.count('\ndef '),
                           'classes': code_text.count(' class ') + code_text.count('\nclass '),
                           'modules': code_text.count(' import ') + code_text.count('\nimport '),
                           'methods': code_text.count('.'),
                           'cycles': code_text.count(' for ') + code_text.count('\nfor '),
                           'conditions': code_text.count(' class ') + code_text.count('\nif '),
                           'operators': code_text.count('+') + code_text.count('-') + code_text.count(
                               '*') + code_text.count('/'),
                           'avg len': self.compute_avg_len(code_text),
                           'avg_density': np.mean([len(row) for row in code_text.split("\n")]) / len(
                               code_text.split("\n")),
                           'cos_sim': cos_sim,
                           'is_plagiat': target
                           }, index=filename)
        return df

    def make_dataset(self, filename1: str, filename2: str, path1: str, path2: str) -> pd.DataFrame:
        final_frames = []
        transformed_origin, transformed_plagiat_1 = self.normalize_code(filename1, path1, path2)
        cos_sim = self.compute_cos_sim(transformed_origin, transformed_plagiat_1)
        df_origin = self.make_df(transformed_origin, [filename1], 1, 0)
        df_plag1 = self.make_df(transformed_plagiat_1, [filename1], cos_sim, 1)
        final_frames.append(df_origin)
        final_frames.append(df_plag1)

        return pd.concat(final_frames)

    def predict_model(self, input_path, model_name, answer_filename):
        with open(model_name, 'rb') as file:
            model = pickle.load(file)

        self.get_paths_and_filenames(input_path)
        index = 0
        for i in range(int(len(self.folders_paths)/2)):
            data = self.make_dataset(self.files_names[index],
                                     self.files_names[index],
                                     self.folders_paths[index],
                                     self.folders_paths[index+1])

            index += 2
            self.answers.append(model.predict_proba(data)[1][1])

        with open(answer_filename, 'w') as f:
            for answer in self.answers:
                f.write(str(answer.round(3)) + '\n')

        return 'Success'


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('input', type=str, help='Input file with codes')
    parser.add_argument('scores', type=str, help='Output file with scores')
    parser.add_argument('--model', type=str, help='Model filename')
    args = parser.parse_args()

    input_path = args.input
    model = Compare(input_path)
    answer_path = args.scores
    pkl_filename = args.model
    print(model.predict_model(input_path, pkl_filename, answer_path))


if __name__ == main():
    main()