# Disaster Response Project
> Machine learning pipeline that classifies messages into categories to better help first responders provide aid when recieving high volumes of messages during a disaster.


## Table of Contents
* [General Info](#general-information)
* [How to Run](#how-to-run)
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
- Motivation: Interested in learning a foundational knowledge of machine learning algorithms. This project provided a great introduction into machine learning libraries and the general process and steps taken in constructing a model from scratch.
- Skills Learned: Debugging, terminal commands, encapsulation of application functionality into multiple modules and machine learning algorithms/libraries.


## How to Run
1. Run the main() function in the process_data.py module - This will extract the data from the csv source files, transform, and load the data to a sqlite data table. Example from root directory: "data/python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db"
3. Run the main() function in the train_classifier.py module - This will use the sqlite data table to train a machine learning model. The model is dumped to a pickle file. Example from root directory: "python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl"
4. Run the terminal command "python run.py" in the app directory to run the web app
5. Go to http://127.0.0.1/


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
![image](https://user-images.githubusercontent.com/70555199/125826316-035a5a3d-17ef-4657-a535-9f26f6712d7d.png)

- Overview of training data set used to train model
![image](https://user-images.githubusercontent.com/70555199/125826492-e0529ec9-a411-45db-81d5-76bdd445c3c7.png)


## Setup
- Anaconda Distribution
- IDE such as Microsoft Visual Studio Code


## Project Status
- Work In Progress (WIP) - Attempting to create virtual environment on cloud platform to better showcase application


## Conclusion
 - Machine learning classification model is a powerful tool that can help first responders classify distress messages into categories. This can help first responders provide aid more effectively and timely in a fast-paced, high stress disaster scenario.


## Room for Improvement
- Hyperparameter tuning in random forest classifier. Due to machine processing limitations I only selected n__estimator parameter for optimization. Model can be improved further.
- Training dataset does not include certain categories. No matter how tuned the model is, it cannot overcome a lack of categorical data in the training dataset.


## Acknowledgements
Project was made possible by Udacity platform and included disaster csv datasets

## Contact
Created by [@mlevanduski]


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->
