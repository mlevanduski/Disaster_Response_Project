import json
import plotly
import pandas as pd
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import sklearn
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load('C:/Users/MLevanduski/Desktop/Coursework/Data Science Udacity/Data_Engineering_Project/disaster_response_pipeline_project/models/classifier.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # First visual - Bar chart of sum of messages by category type
    # Extracting column names and sum of message classifications per category
    cat_labels = []
    cat_sum = []
    for col in df.columns[4:]:
        cat_labels.append(col)
        cat_sum.append(df[col].sum())

    # Creating a dictionary to sort category name and value pairs
    cat_dict = dict(zip(cat_labels, cat_sum))
    cat_list_tuples = sorted(cat_dict.items(), key=lambda x:x[1],reverse=True)

    # Unpacking list of tuples in order to plot
    x_labels, y_values = map(list, zip(*cat_list_tuples))

    # Second visual - Pie chart of genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = [
        # Graph one bar chart of sum of messages by category type
        {
            'data': [
                Bar(
                    x=x_labels[:5],
                    y=y_values[:5]
                )
            ],

            'layout': {
                'title': 'Top 5 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },

        # Graph 2 Pie chart of % composition of message genres
        {
            'data': [
                Pie(
                    labels = genre_names,
                    values = genre_counts
                )
            ],

            'layout': {
                'title': 'Pie Chart of Message Genres',
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
    app.run(host='127.0.0.1', port=3001, debug=True)

if __name__ == '__main__':
    main()