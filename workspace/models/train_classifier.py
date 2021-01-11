# import libraries
import sys

# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import os
from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """ 
    Load data from DisasterResponse database by sqlite method 
    
    Args:
    database_filepath: string. The string that wants to do word tokenization.
    
    Returns:
    X: The features pandas dataframe.
    Y: The target pandas dataframe.
    category_names: category names.
    
    """
    # Connect to database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Fetch dataset from database
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Select the features (X) and target (Y)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    
    # Fetch category names
    category_names = Y.columns.values
    
    return X, Y, category_names


def tokenize(text):
    """ 
    Tokenize and lemmatize word in a string
    
    Args:
    text: string. The string that wants to tokenize.
    
    Returns:
    clean_tokens: The text after tokenize and lemmatize word in a string
    
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """ 
    Perform the data pipeline. Then search for the best model parameters by using grid-search cross-validation
    
    Returns:
    cv: The model that includes best model parameters.
    
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'features__text_pipeline__vect__max_features': (None, 5000)
        'clf__estimator__n_estimators': [100, 150]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv 

def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Evaluate model.
    Print the model evaluation.
     
    Args:
    model: object. The model object.
    X_test: dataframe. The feature pandas dataframe.
    Y_test: dataframe. The label/target pandas dataframe.
    category_names: object. The strings of category names.
    
    """
    # predict on test data
    y_pred = model.predict(X_test)

    for i, predict in enumerate(y_pred):
        print(classification_report(Y_test.iloc[i], predict))


def save_model(model, model_filepath):
    """ 
    Save the model into pickle file
    
    Args:
    model: object. The model.
    model_filepath: string. The path that you want to save the model.
    
    """
    # Save model to model filepath by using pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ 
    Load, Buid data and then train the model. Then save the model to pickle 
   
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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