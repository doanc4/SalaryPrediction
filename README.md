# Contents
The main file in this repository is Salary Prediction and NLP.ipynp. This is the Jupyter notebook that contains all of the modeling and analysis. The Notebook depends on the other files in the repo, for example the several .py files, which contain code used in the analyses. 
- salary_prediction.py contains the main predictive model class
- preprocessing.py contains preprocessing functions
- nlp.py contains NLP-related functions
- runbert.py contains the code used to create BERT text embeddings in the cloud

The model_artifacts directory contains serialized artifacts needed to load the trained model.

The .pkl files are simply there for convenience in the notebook. 
# Setting up the the virtual environment
This virtual environment was set up using pipenv, so if you have pipenv installed I would highly recommend installing the virtual environment dependencies by cd'ing into this directory and running "pipenv install". This will install all the dependencies listed in the Pipfile and version them properly according to the Pipfile.lock. If you run into issues locking, you can run "pipenv install --skip-lock" which will install versions of the dependencies that will work for you, but may not be the same as the ones I used.

If you do not want to use pipenv, you can install the dependencies from the requirements.txt file. 