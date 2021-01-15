# Capstone Project: Azure Machine Learning Engineer

The Capstone Project comes as the last project within the Udacity Azure Machine Learning nano degree. During this project we will create two models, The first one using Automated ML and the other one using hyperparameters. Finally compare the two models and deploy the best performing model.  

## Dataset

### Overview
In regards with the dataset, It's a collection if 10minutes of data over nine-thousand unique games. The games are all ranked games at high elo (Diamond 1 to Masters). This dataset contains 40 columns including the "gameId" column which is a unique identifier. The target column is called "blueWins". The blue team wins the gain if "blueWins" is 1, otherwise the red team wins. The source of the dataset is https://www.kaggle.com/ where multiple dataset has been published for free to use in experiments and learning projects 

### Task
The purpose of this project is to build an accurate machine learning model that can determine whether a team will win a game based on the first 10 minutes of the game. From this insight, we can then look at the characteristics of the winning team. Basically we need to predict a boolean typed value 0 or 1 which means that this is a classification problem.

### Access
The dataset has been uploaded to the Azure ML studio and registered from this repository "https://raw.githubusercontent.com/ahmedkhammessi/nd00333-capstone/master/starter_file/high_diamond_ranked_10min.csv". On the Jupyter notebook level, I used TabularDatasetFactory class to get the csv file.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
