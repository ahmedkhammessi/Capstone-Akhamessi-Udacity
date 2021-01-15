from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

url = "https://raw.githubusercontent.com/ahmedkhammessi/nd00333-capstone/master/starter_file/high_diamond_ranked_10min.csv"
ds = TabularDatasetFactory.from_delimited_files(url)


#Save model for current iteration

run = Run.get_context()
  
x_df = ds.to_pandas_dataframe().dropna()

y_df = x_df.pop("blueWins")

# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=123)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=100)

    args = parser.parse_args()

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    run.log("Regularization:", np.float(args.C))
    run.log("Maximum Iterations:", np.int(args.max_iter))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperDrive_{}_{}'.format(args.C,args.max_iter))

if __name__ == '__main__':
    main()