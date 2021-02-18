# import libraries
import sys

import pandas as pd
import numpy as np

import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Load data from DisasterResponse database by sqlite method.
    Then merge two datasets into a pandas dataframe.
    
    Args:
    messages_filepath: string. The path of messages dataset.
    categories_filepath: string. The path of categories dataset.
    
    Returns:
    df: The dataframe that contains messages and categories data.
    
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer')
    
    return df

def clean_data(df):
    """ 
    Clean the dataset
    
    Args:
    df: dataframe. The pandas dataframe that wants to clean.
    
    Returns:
    cleaned_df: The cleaned dataframe.
    
    """
    
    ## Split categories into separate category columns
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";", expand=True)
    categories.head()
    
    # Select the first row of the categories dataframe
    row = categories.iloc[:1]

    # Extract a list of new column names for categories.
    category_colnames = row.apply(lambda row: row[0][0:-2],axis=0)
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()   
    
    ## Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda categories: categories[-1])

        ## Convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)
    
    # Convert 2 values of related column to 1 because the 2 values related to disaster likewise 1 values 
    # Except that the 2 value contains an incomplete, wrong pattern or other languages than English response messages. 
    categories['related'].replace(2, 1, inplace=True)
    
    ## Replace categories column in df with new category columns
    # Drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    ## Remove duplicates
    cleaned_df = df.drop_duplicates()
    
    return cleaned_df


def save_data(df, database_filepath, database_filename):
    """ 
    Clean the dataset.
    
    Args:
    df: dataframe. The pandas dataframe that wants to clean.
    database_filepath: The directory of database that you wants to save.
    database_filename: The database file.
    
    """
    # Connect to database
    #filepath = 'sqlite:///' + database_filepath
    #engine = create_engine(filepath)
    engine = create_engine('sqlite:///' + database_filepath)

    # Convert dataframe to sql then save it to database 
    df.to_sql(database_filename, engine, index=False)


def main():
    """ 
    Load and clean the dataset then saves it to database.
    
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('sqlite:///' + database_filepath)
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, 'DisasterResponse')
        
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