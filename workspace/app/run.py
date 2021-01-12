import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine

import seaborn as sns

app = Flask(__name__)

def tokenize(text):
    """ 
    Tokenize and lemmatize word in a string
    
    Args:
    text: string. The string that wants to tokenize.
    
    Returns:
    clean_tokens: The text after tokenize and lemmatize word in a string
    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ 
    Render the home (index) webpage  
    
    Returns:
    render_template: the Rendered web page with plotly graphs.
    
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = df.drop(['id','message','original','genre'],axis=1)
    category_names = categories.columns
    category_sum = categories.sum().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=category_names,
                    values=category_sum
                )
                
            ],
            'layout': {
                'title': 'The distribution of Message Categories'
           
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hole=0.3
                )
                
            ],

            'layout': {
                'title': 'Message Genres Pie chart'
           
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
                
            ],

            'layout': {
                'title': 'The distribution of Message Genres'
           
            }
        }
        
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    """ 
    Render the go (predict) webpage  
    
    Returns:
    render_template: the Rendered go(predict) web page with the predict result.
    
    """
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """ 
    Run the web application 
   
    """
        
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()