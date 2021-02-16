# Disaster response analysis
### Table of Contents 
1. [Project Motivation](#Project-Motivation)
2. [Description](#Description)
3. [Instructions](#Instructions)
4. [Libraries](#Libraries)
5. [File Descriptions](#File-Descriptions)
6. [Results](#Results)

## Project Motivation
The disaster response messages are the crucial message that can be 

## Description
Analyzing data for disaster response messages by classifying data into several categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Libraries
* sqlalchemy
* flask
* pandas
* numpy
* json
* sklearn
* os
* sys
* nltk
* re
* plotly
* seaborn

## File Descriptions
```app/```: The application folder. <br/>
```app/static/```: The CSS file. <br/>
```app/templates/```: The html file. <br/>
```app/run.py```: The file where the code are presented. <br/>
```data/```: The folder that contains dataset. <br/>
```data/DisasterResponse.db```: The Disaster-Response database. <br/>
```data/disaster_categoroes.csv```: The disaster categories dataset. <br/>
```data/disaster_messages.csv```: The disaster messages dataset. <br/>
```data/process_data.csv```: The code of preprocessing data. <br/>
```models/```: The folder that contains classifier model. <br/>
```models/classifier.pkl```: The classifier model in pickle format. <br/>
```models/train_classifier.py```: The code of training classifier. <br/>
```readme.md```: README file. <br/>

## Results
The prediction of message categories 

