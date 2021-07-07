import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    # Create engine to read in sql database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', con = engine)
    
    # Replace any 2 values in matrix with 1 for model classification to work
    df.replace(2, 1, inplace = True)
    
    # Initialize samples matrix and target values matrix
    X = df['message'].values # Input message
    dfy = df.drop(columns = ['id','message','original','genre'])
    Y = dfy.values # Categories that input message can be classified as, can have multiple categories per message

    # Rename columns by category names
    category_names = dfy.columns.values

    return X, Y, category_names


def tokenize(text):
    # Remove Punctuation
    text = re.sub(r'[^\w\s]','',text)
    # Tokenize message
    words = word_tokenize(text)
    # Lemmatizer initialize
    lemmatizer = WordNetLemmatizer()
    # Remove stop words from tokens
    words = [w for w in words if w not in stopwords.words("english")]
    clean_words = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).lower().strip()
        clean_words.append(clean_word)

    return clean_words


def build_model():
    # Build a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
    ])

    # Set a hyperparameter to optimize using gridsearch. I will set the number of tress in the random forest model n_estimators
    parameters = {
        'clf__estimator__n_estimators': [5,10,15]    
    }
    
    # Funnel pipeline into gridsearch object
    gs_object = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return gs_object
    

def evaluate_model(model, X_test, y_test, category_names):
    # Predict message categories
    y_pred = model.predict(X_test)
    
    # Output classification report
    report = classification_report(y_test, y_pred, target_names = category_names)

    return report

def save_model(model, model_filepath):
    # Dump model to a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))

    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()