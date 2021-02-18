# Disaster response analysis
### Table of Contents 
1. [Project Motivation](#Project-Motivation)
2. [Description](#Description)
3. [Instructions](#Instructions)
4. [Libraries](#Libraries)
5. [File Descriptions](#File-Descriptions)
6. [Results](#Results)

## Project Motivation
The disaster response messages are the crucial message that needs time to classify because they are sensitive. If the process can be done by automatic, it could save a lot of time for emergency work. Therefore, I want to solve this problem to reduce the workload of the emergency response officers.

## Description
Analyzing and classified disaster response messages by natural language processing (NLP). Then shows the data visualization of results on the website by Flask framework.

## Instructions
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
```workspace/app/```: The application folder. <br/>
```workspace/app/static/```: The CSS file. <br/>
```workspace/app/templates/```: The html file. <br/>
```workspace/app/run.py```: The file where the code are presented. <br/>
```workspace/data/```: The folder that contains dataset. <br/>
```workspace/data/DisasterResponse.db```: The Disaster-Response database. <br/>
```workspace/data/disaster_categoroes.csv```: The disaster categories dataset. <br/>
```workspace/data/disaster_messages.csv```: The disaster messages dataset. <br/>
```workspace/data/process_data.csv```: The code of preprocessing data. <br/>
```workspace/models/```: The folder that contains classifier model. <br/>
```workspace/models/classifier.pkl```: The classifier model in pickle format. <br/>
```workspace/models/train_classifier.py```: The code of training classifier. <br/>
```readme.md```: README file. <br/>

## Results
The model classifier can predict disaster response catergories. Note that there are 36 catergories that can predict. 
