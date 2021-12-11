# Disaster Response Pipeline Project
This is a web app backed with a machine learning model to classify a textual message into one or more disaster categories.

### Main dependencies:
* Pandas
* Numpy
* sqlalchemy
* sklearn
* NLTK
* Flask
* Plotly


## Project structure:
* **app**: the web app files.
    * __templates:__ view templates for web pages.
    * __run.py:__ a python script to start the Flask server.
* **data**: the datasets directory.
    * __disaster_categories.csv:__ a csv file that contains the messages cathories for each message id.
    * __disaster_messages.csv:__ a csv file that contains message id, text, and genre.
    * __DisasterResponse.db:__ an sqlite database used by the web app at runtime.
    * __process_data:__ a python script to generate the sqlite database file from updated csv file.
* **models**: the classification models directory
    * __classifier.pkl:__ an exported multioutput message classifier model to be used by the web app at runtime.
    * __train_classifier.py:__ a python script to build, train, and export the classifier model based on updated data.
    
### Instructions:
1. Run the following commands in the project's root directory to set the database and classifier model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/
