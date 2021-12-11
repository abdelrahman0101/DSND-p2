import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sqlalchemy import create_engine
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from time import process_time
import pickle

lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    nltk.download(['punkt', 'wordnet', 'stopwords'])
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    labels = list(Y.columns)
    return X, Y, labels

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000, solver='sag')))
    ])
    parameters = {
        'clf__estimator__C': [1.0, 5.0, 10.0]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pred = model.predict(X_test)
    f1_scores = []
    acc_scores = []
    for c in range(0, pred.shape[1]):
        print(category_names[c].capitalize());
        print(classification_report(Y_test[category_names[c]], pred[:, c]))
        acc_scores.append(accuracy_score(Y_test[category_names[c]], pred[:, c]))
        print("Accuracy:", acc_scores[-1])
        f1_scores.append(f1_score(Y_test[category_names[c]], pred[:, c], average='macro'))
        print("Macro F1-score", f1_scores[-1])
        print("________________________________")
    print("Average Macro F1 score:", np.average(f1_scores))
    print("Average Accuracy:", np.average(acc_scores))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        t_start = process_time()
        model.fit(X_train, Y_train)
        t_end = process_time()
        print("Training time: %02d:%02.2d" % ((t_end-t_start)/60, (t_end-t_start)%60));
        
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