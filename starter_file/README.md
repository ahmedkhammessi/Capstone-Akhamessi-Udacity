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
The Auto ML settings and configuration used for this experiment are respectevly the following
- experiment_timeout_minutes : 20   --> Defines a termination or an exit criterea
- max_concurrent_iterations: 5  --> The max number of iteration that could be executed in parallel
- primary_metric : 'accuracy'   --> The primary metric that the Auto ML will try to optimize
- task='classification' --> The type of the task that Auto ML will try to solve and it could be other values such as Regression
- training_data=ds  --> The Dataset that the Auto ML will work with.
- label_column_name='blueWins'  --> The name of the column that the algorithm is trying to predict
- n_cross_validations=5 --> The number of cross validation to execute

### Results
The result of the experiment showed that the `VotingEnsemble` algorithm trained the best model with an accuracy level of : 0.7336771895659304 . Below we can see the exhaustive list of the parameters used for the prefitted model:
prefittedsoftvotingclassifier
{'estimators': ['40', '55', '20', '19', '53', '38', '33', '68', '48', '28'],
 'weights': [0.18181818181818182,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091]}

40 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': True,
 'with_std': False}

40 - lightgbmclassifier
{'boosting_type': 'goss',
 'class_weight': None,
 'colsample_bytree': 0.5944444444444444,
 'importance_type': 'split',
 'learning_rate': 0.015797894736842105,
 'max_bin': 270,
 'max_depth': 7,
 'min_child_samples': 750,
 'min_child_weight': 0,
 'min_split_gain': 0.3684210526315789,
 'n_estimators': 200,
 'n_jobs': 1,
 'num_leaves': 158,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.6842105263157894,
 'reg_lambda': 0.8421052631578947,
 'silent': True,
 'subsample': 1,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

55 - maxabsscaler
{'copy': True}

55 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 0.99,
 'importance_type': 'split',
 'learning_rate': 0.06316157894736842,
 'max_bin': 100,
 'max_depth': -1,
 'min_child_samples': 886,
 'min_child_weight': 7,
 'min_split_gain': 0.7368421052631579,
 'n_estimators': 25,
 'n_jobs': 1,
 'num_leaves': 206,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.631578947368421,
 'reg_lambda': 0.631578947368421,
 'silent': True,
 'subsample': 0.29736842105263156,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

20 - minmaxscaler
{'copy': True, 'feature_range': (0, 1)}

20 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 0.2977777777777778,
 'importance_type': 'split',
 'learning_rate': 0.015797894736842105,
 'max_bin': 360,
 'max_depth': 2,
 'min_child_samples': 750,
 'min_child_weight': 3,
 'min_split_gain': 0.3157894736842105,
 'n_estimators': 200,
 'n_jobs': 1,
 'num_leaves': 176,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.15789473684210525,
 'reg_lambda': 0.7894736842105263,
 'silent': True,
 'subsample': 0.99,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

19 - maxabsscaler
{'copy': True}

19 - extratreesclassifier
{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 0.7,
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.035789473684210524,
 'min_samples_split': 0.01,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 400,
 'n_jobs': 1,
 'oob_score': True,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}

53 - maxabsscaler
{'copy': True}

53 - lightgbmclassifier
{'boosting_type': 'goss',
 'class_weight': None,
 'colsample_bytree': 0.99,
 'importance_type': 'split',
 'learning_rate': 0.07368684210526316,
 'max_bin': 90,
 'max_depth': 7,
 'min_child_samples': 614,
 'min_child_weight': 3,
 'min_split_gain': 0.2631578947368421,
 'n_estimators': 50,
 'n_jobs': 1,
 'num_leaves': 227,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.21052631578947367,
 'reg_lambda': 0,
 'silent': True,
 'subsample': 1,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

38 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': True,
 'with_std': False}

38 - lightgbmclassifier
{'boosting_type': 'goss',
 'class_weight': None,
 'colsample_bytree': 0.6933333333333332,
 'importance_type': 'split',
 'learning_rate': 0.026323157894736843,
 'max_bin': 190,
 'max_depth': 8,
 'min_child_samples': 239,
 'min_child_weight': 1,
 'min_split_gain': 0.10526315789473684,
 'n_estimators': 50,
 'n_jobs': 1,
 'num_leaves': 197,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.7368421052631579,
 'reg_lambda': 0.3684210526315789,
 'silent': True,
 'subsample': 1,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

33 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

33 - lightgbmclassifier
{'boosting_type': 'goss',
 'class_weight': None,
 'colsample_bytree': 0.8911111111111111,
 'importance_type': 'split',
 'learning_rate': 0.0842121052631579,
 'max_bin': 220,
 'max_depth': 7,
 'min_child_samples': 205,
 'min_child_weight': 5,
 'min_split_gain': 1,
 'n_estimators': 25,
 'n_jobs': 1,
 'num_leaves': 143,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.47368421052631576,
 'reg_lambda': 0.05263157894736842,
 'silent': True,
 'subsample': 1,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

68 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

68 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 0.4955555555555555,
 'importance_type': 'split',
 'learning_rate': 0.026323157894736843,
 'max_bin': 160,
 'max_depth': 4,
 'min_child_samples': 477,
 'min_child_weight': 2,
 'min_split_gain': 0.2631578947368421,
 'n_estimators': 25,
 'n_jobs': 1,
 'num_leaves': 251,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.7894736842105263,
 'reg_lambda': 0.3684210526315789,
 'silent': True,
 'subsample': 0.3963157894736842,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

48 - robustscaler
{'copy': True,
 'quantile_range': [25, 75],
 'with_centering': False,
 'with_scaling': False}

48 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 0.7922222222222222,
 'importance_type': 'split',
 'learning_rate': 0.06842421052631578,
 'max_bin': 140,
 'max_depth': 10,
 'min_child_samples': 375,
 'min_child_weight': 0,
 'min_split_gain': 0.9473684210526315,
 'n_estimators': 25,
 'n_jobs': 1,
 'num_leaves': 83,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.6842105263157894,
 'reg_lambda': 0.9473684210526315,
 'silent': True,
 'subsample': 0.6931578947368422,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

28 - maxabsscaler
{'copy': True}

28 - extratreesclassifier
{'bootstrap': False,
 'ccp_alpha': 0.0,
 'class_weight': 'balanced',
 'criterion': 'entropy',
 'max_depth': None,
 'max_features': 'sqrt',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.06157894736842105,
 'min_samples_split': 0.33789473684210525,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 600,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}

![alt text](https://github.com/ahmedkhammessi/nd00333-capstone/blob/master/starter_file/screens/rundetails_automl_notebook.PNG)

The best model

![alt text](https://github.com/ahmedkhammessi/nd00333-capstone/blob/master/starter_file/screens/bestmodel_runid.PNG)

As an improvement for this experience with Auto ML we could extend the running time to get more precise results but the tradeoff is the costs of course. Also using different primary metrics such as AUC and going for the ' entire grid' option which we go through the exhaustive list of columns but it will give a more accurate result.

## Hyperparameter Tuning

Logistic regression is a pretty simple—yet very powerful—algorithm used in data science and machine learning. It is a statistical algorithm that classifies data by considering outcome variables on extreme ends and creates a logarithmic line to distinguish between them. In the typical case, you’re trying to decide for a given data point whether a predicate is true or false which is exactly our task we want to predict the value of the `blueWins` column. In regards with the parameters I used for the maximum number of iterations taken for the solvers to converge `max-iter` the following array [40,80,120,130,200] and as for the Inverse of regularization strength `C` the following array [1,2,3,4].

As a best practice and in order to avoid waisting time and money I included the use of `BanditPolicy` which is an early termination policy if the primary metric is not within the acceptable range.


### Results
The tuned hyperparameter via hyperdrive are `Regularization Strength` : 2.0 and `Max Iterations`: 200 which achieved an accuracy level of 0.729757085020243 

![alt text](https://github.com/ahmedkhammessi/nd00333-capstone/blob/master/starter_file/screens/hyperparam-rundetails.PNG)

and the best model is

![alt text](https://github.com/ahmedkhammessi/nd00333-capstone/blob/master/starter_file/screens/hyperparam-bestmodel.PNG)

The improvments that we can work on are first of all the primary metric cause most probably the Auccracy is not the most appropriate for the model and we can replace the `Classification` with the Bayesian Parameter Sampling instead of Random. Bayesian sampling tries to intelligently pick the next sample of hyperparameters, based on how the previous samples performed.

## Model Deployment

So far we have two trained models, One delivered by Auto ML and the other by Hyperparameter tuning. In our case we use the most accurate which is the Auto ML generated model which for it's deployment will require the inference configuration including the scoring and environment script and configure the deployment such the choice of the platform AKS or container instances CPU or GPU and so on.

To Demo the deployed model we will send json paylod and explore the response as shown in the screenshot

![alt text](https://github.com/ahmedkhammessi/nd00333-capstone/blob/master/starter_file/screens/deployedmodel-test.PNG)

This the deployed model in a Healthy state

![alt text](https://github.com/ahmedkhammessi/nd00333-capstone/blob/master/starter_file/screens/active-model.PNG)

## Screen Recording
https://youtu.be/013VjKdpxAc
