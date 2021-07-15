# Disaster Response Project
> Machine learning pipeline that classifies messages into categories to better help first responders provide aid when recieving high volumes of messages during a disaster.


## Table of Contents
* [General Info](#general-information)
* [Files](#files)
* [Technologies Used](#technologies-used)
* [Libraries Used](#libraries-used)
* [Features](#features)
* [Setup](#setup)
* [Project Status](#project-status)
* [Conclusion](#conclusion)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- Motivation: Interested in learning more about nightly price influences and trends in Boston Airbnb market


## Files
- disaster_categories.csv - csv file that houses categorical message data used in training set
- disaster_messages.csv - csv file that houses message body data used in training set
- process_data.py - python3 file that extracts, transforms and loads data from csv files and returns a sqlite database
- train_classifier.py - python3 file that accesses the sqlite database from process_data.py and trains a random forest model (utilizing MultiOutputClassifier) that can predict message category. Returns a pickle file of trained model
- run.py - python3 file that utilizes Flask as a back-end web framework
- go.html - html file for web app front end displayed when user enters and submits message
- master.html - html file for web app front end displaying plotly graphs prior to user input

## Technologies Used
- Python3
- JupyterLab - version 2.1.5
- HTML
- sqlite


## Libraries Used
- pandas
- sklearn
- numpy
- re
- sqlalchemy
- nltk
- plotly
- json
- flask


## Features
- Message Classifier

- Overview of training data set used to train model


## Setup
Anaconda Distribution


## Project Status
Complete


## Conclusion



## Room for Improvement
- Hyperparameter tuning in random forest classifier. Due to machine processing limitations only selected n__estimator parameter for tuning. Model can be improved further.
- Training dataset does not include certain categories. No matter how tuned the model is, it cannot overcome a lack of categorical data in the training dataset.


## Acknowledgements
Project was made possible by Udacity platform and included disaster csv datasets

## Contact
Created by [@mlevanduski]


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->
