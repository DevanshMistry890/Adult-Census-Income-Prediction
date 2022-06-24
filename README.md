# Adult-Census-Income-Prediction
Internship Project

Project Title: Adult Census Income Prediction
Technologies:  Machine Learning Technology
Domain:  Finance
Project Difficulties level:  Intermediate
Problem Statement:
The Goal is to predict whether a person has an income of more than 50K a year or not.
This is basically a binary classification problem where a person is classified into the 
>50K group or <=50K group.

# Note:
content here is not safe to use in any project, i am will be not resopnsible for any kind of copyright issues. I am using this repo for just my ease.


## ML-Model-Flask-Deployment
This project is to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict employee salaries absed on trainign data in 'hiring.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

Enter valid numerical values in all 4 input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary vaule on the HTML page!
