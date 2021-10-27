import numpy as np
import pandas as pd
from nlp import clean_text


def get_bin(salary, percentiles):
    """Bins salaries into groups based on percentile. Used for fine-tuning BERT."""
    for i, val in enumerate(percentiles):
        if i < len(percentiles) - 1:
            if salary > val and salary <= percentiles[i + 1]:
                bin_label = i
                break
        else:
            bin_label = i
    return bin_label


def preprocess_job(job_object, dummy_cols, categorical_vars, vectorizer, lemmatizer):
    """Data preprocessing function to get input data for a novel job posting into the format required by the regression model.
    Params:
    job_object: a dict whose key-value pairs are property names and values of a given job posting.
    dummy_cols: a list specifying the column names associated with the trained model.
    categorical_vars: a list specifying the categorical variables used for the model.
    vectorizer: the text vectorizer object used to transform the text variables into vectors.
    """
    assert all(var in job_object for var in categorical_vars), f"All categorical variables must be present in the Job Object. See: {categorical_vars}"
    assert 'Title' in job_object, "Job object must contain a 'Title' feature"
    dummy_dict = {dummy_col: [] for dummy_col in dummy_cols}
    for k, v in job_object.items():
        for dummy in dummy_dict:
            if dummy.startswith(f'{k}_'):
                if dummy.endswith(v):
                    dummy_dict[dummy] = 1
                else:
                    dummy_dict[dummy] = 0
                    
    clean_title = clean_text(job_object['Title'], lemmatizer=lemmatizer)
    title_vec = vectorizer.transform([clean_title])
    title_vec = np.array(title_vec.todense()).flatten()
    features = vectorizer.get_feature_names_out()
    title_vec = pd.Series(title_vec, index=features)
    features = pd.concat([pd.Series(dummy_dict), title_vec], axis=0)
    return features
