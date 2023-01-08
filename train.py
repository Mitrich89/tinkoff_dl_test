import argparse
import os
import numpy as np
import pathlib
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from normalizer import AstTransformer
from catboost import CatBoostClassifier as cb


class Train:
    def __init__(self, files_path, plagiat1_path, plagiat2_path):
        self.files_path = files_path
        self.plagiat1_path = plagiat1_path
        self.plagiat2_path = plagiat2_path


    def normalize_code(self, file_name):
        transformer = AstTransformer()
        transformed_origin = transformer.transform_file(pathlib.Path(f'{self.files_path}/{file_name}'))
        transformed_plagiat_1 = transformer.transform_file(pathlib.Path(f'{self.plagiat1_path}/{file_name}'))
        transformed_plagiat_2 = transformer.transform_file(pathlib.Path(f'{self.plagiat2_path}/{file_name}'))
        return transformed_origin, transformed_plagiat_1, transformed_plagiat_2

    def compute_cos_sim(self, origin, plagiat) -> float:
        pair = [origin, plagiat]  # создаем пару из уже нормализованных кодов
        vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_vectors = vectorizer.fit_transform(pair)  # векторизируем с помощью TF-IDF
        return cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

    def compute_avg_len(self, code_text:str) -> float:
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

    def make_dataset(self, path:str) -> pd.DataFrame:
        final_frames = []
        for file_name in os.listdir(path):
            try:
                transformed_origin, transformed_plagiat_1, transformed_plagiat_2 = self.normalize_code(file_name)
                cos_sim_1 = self.compute_cos_sim(transformed_origin, transformed_plagiat_1)
                cos_sim_2 = self.compute_cos_sim(transformed_origin, transformed_plagiat_2)

                df_origin = self.make_df(transformed_origin, [file_name], 1, 0)
                df_plag1 = self.make_df(transformed_plagiat_1, [file_name], cos_sim_1, 1)
                df_plag2 = self.make_df(transformed_plagiat_2, [file_name], cos_sim_2, 1)

                final_frames.append(df_origin)
                final_frames.append(df_plag1)
                final_frames.append(df_plag2)
            except Exception as _:
                pass

        return pd.concat(final_frames)

    def train_model(self, modelname:str) -> str:
        dataset = self.make_dataset('files')
        X, y = dataset.drop(columns=['is_plagiat']), dataset['is_plagiat']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.65, random_state=42)
        model = cb(objective='CrossEntropy',
                   colsample_bylevel=0.06610164865527374,
                   depth=12,
                   boosting_type='Plain',
                   bootstrap_type='Bayesian',
                   bagging_temperature=7.298206489243199)
        model.fit(X_train, y_train)
        with open(modelname, 'wb') as file:
            pickle.dump(model, file)

        return 'Success'


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('files', type=str, help='Input dir for original codes')
    parser.add_argument('plagiat1', type=str, help='Input dir for plagiarism codes')
    parser.add_argument('plagiat2', type=str, help='Input dir for plagiarism codes')
    parser.add_argument('--model', type=str, help='Model filename')
    args = parser.parse_args()

    files_path = args.files
    plagiat1_path = args.plagiat1
    plagiat2_path = args.plagiat2
    model = Train(files_path, plagiat1_path, plagiat2_path)
    pkl_filename = args.model
    result = model.train_model(pkl_filename)
    print(result)


if __name__ == "__main__":
    main()


