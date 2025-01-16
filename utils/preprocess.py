# utils/preprocess.py
import pandas as pd

def preprocess_input(title, abstract, vectorizer):
    """
    Preprocess user input before prediction.
    :param title: str, title of the paper
    :param abstract: str, abstract of the paper
    :param vectorizer: TfidfVectorizer, trained vectorizer
    :return: array, processed input data
    """
    combined_text = title + " " + abstract
    input_vector = vectorizer.transform([combined_text]).toarray()
    return input_vector
