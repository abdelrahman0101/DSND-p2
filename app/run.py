import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Tokenize the specified text and return a list of clean tokens"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    The app's homepage
    """
    # extract data needed for visuals
    category_counts = df.drop(['id', 'genre', 'message', 'original'], axis=1).sum(axis=0)
    category_names = list(category_counts.index)

    correlations = df[category_names].corr().values.tolist();
    for i in range(len(correlations)):
        correlations[i][i] = None;

    # Plotting the entire correlation matrix is a lot of details in a single graph, so we plot a single category
    death_corr = correlations[category_names.index('death')]
    del death_corr[category_names.index('death')] # no need to show a redundant correlation

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': {
                    'text': "Number of messages in each category",
                    'font': {
                        'color': 'grey',
                        'size': 24
                    }
                },
                'yaxis': {
                    'title': {
                        'text': "Count",
                        'font': {
                            'color': 'grey',
                            'size': 18
                        }
                    },
                },
                'xaxis': {
                    'title': {
                        'text': "Category",
                        'font': {
                            'color': 'grey',
                            'size': 18
                        }
                    },
                    'tickangle': 45
                },
                'margin': {
                    'b': 130,
                },
                'height': 600,
            }
        },
        {
            'data': [
                Bar(x=[c for c in category_names if c != 'death'], y=death_corr)
            ],
            'layout': {
                'title': {
                    'text': "Correlations of 'Death' with other categories",
                    'font': {
                        'color': 'grey',
                        'size': 24
                    }
                },
                'yaxis': {
                    'title': {
                        'text': "Correlation",
                        'font': {
                            'color': 'grey',
                            'size': 18
                        }
                    },
                },
                'xaxis': {
                    'title': {
                        'text': "Category",
                        'font': {
                            'color': 'grey',
                            'size': 18
                        }
                    },
                    'tickangle': 45
                },
                'margin': {
                    't': 170,
                    'b': 130
                },
                'height': 500,
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
    process user query and display classification results
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
    app.run(host='localhost', port=3001, debug=True)


if __name__ == '__main__':
    main()