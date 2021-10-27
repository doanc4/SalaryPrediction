import pickle
import os
from preprocessing import preprocess_job


class SalaryModel:
    def __init__(self, model_dir):
        """Loads the serialized model artefacts from the specified directory. Requires 4 artifacts:
        categorical_vars.pkl: list containing names of categorical variables used to train the model
        dummy_cols.pkl: list containing names of dummy variables used by trained model
        random_forest.pkl: the trained sklearn RandomForestRegressor object
        vectorizer.pkl: the trained sklearn TfidfVectorizer object"""
        categorical_var_path = os.path.join(model_dir, 'categorical_vars.pkl')
        dummy_cols_path = os.path.join(model_dir, 'dummy_cols.pkl')
        random_forest_path = os.path.join(model_dir, 'random_forest.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

        self.cat_vars = pickle.load(open(categorical_var_path, 'rb'))
        self.dummy_cols = pickle.load(open(dummy_cols_path, 'rb'))
        self.rf = pickle.load(open(random_forest_path, 'rb'))
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))

    def preprocess(self, job_object, lemmatizer):
        """Preprocesses a dict containing key-value pairs of job properties into the appropriate features required by the model.
        Params:
        job_object: a dict whose key-value pairs are property names and values of a given job posting.
        lemmatizer: an NLTK WordNetLemmatizer object used to lemmatize text.
        """
        features = preprocess_job(job_object, self.dummy_cols, self.cat_vars, self.vectorizer, lemmatizer)
        return features

    def predict(self, feature_vector):
        """The main salary prediction function. Takes a preprocessed feature vector and returns a predicted salary."""
        return self.rf.predict([feature_vector])        

