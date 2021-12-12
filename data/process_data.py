import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories from external csv files into a joint dataframe
    :param messages_filepath: messages csv file path
    :param categories_filepath: categories csv file path
    :return:
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages_df.join(categories_df.set_index('id'), on='id', how='inner')
    return df
    

def clean_data(df):
    """
    cleans the raw data before it's saved in a database
    :param df: dataframe of raw data
    :return: cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # use the first row to extract a list of new column names for categories.
    row = categories.iloc[0]
    category_colnames = row.apply(lambda s: s[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        # remove classes with no samples in the dataset
        if categories[column].sum() == 0:
            categories = categories.drop([column], axis=1);
            category_colnames = category_colnames[category_colnames != column]
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # drop any row with non-binary category values
    df = df.drop(df[(df[category_colnames] > 1).any(axis=1)].index, axis=0)
    return df


def save_data(df, database_filename):
    """
    Saves the specified dataframe into the specified sqlite database file
    :param df: cleaned dataframe
    :param database_filename: sqlite database file to save the data into
    :return:
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    """
    Script entry point. Parses command line arguments and calls other functions to load data, clean them,
    and store them in an Sqlite database file.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

        
if __name__ == '__main__':
    main()