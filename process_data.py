import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Construct dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, how='inner')
    
    return df
    

def clean_data(df):
    # Create dataframe of 36 elements in category column
    categories = df['categories'].str.split(pat=';', n=0, expand = True)

    # Extract column names from first row of categories dataframe
    category_colnames = [col[:-2] for col in categories.iloc[0]]

    # Rename columns of categories dataframe
    categories.columns = category_colnames    

    # Convert category values to numeric values of either 0 or 1
    for column in categories:
        # Set each value to be the last element in the list of split string elements (numbers)
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str.split('-').str[-1]
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the original categories column from df
    df.drop(columns = ['categories'], inplace = True)

    # Concatenate the original df with the categories df
    cleaned_df = pd.concat([df,categories], axis = 1)
    
    # Replace any 2 values in matrix with 1 for model classification to work
    cleaned_df.replace(2, 1, inplace = True)
    # Remove duplicates
    cleaned_df.drop_duplicates(inplace = True)

    return cleaned_df


def save_data(df, database_filename):
    # Create engine to interface with SQL database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, if_exists = 'replace', index = False)  


def main():
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