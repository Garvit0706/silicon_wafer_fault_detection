from src.exception import CustomException
from src.logger import logging
import sys,os
import boto3
import dill
import numpy as np
import pandas as pd

from pymongo import MongoClient

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split




def export_collection_as_dataframe(db_name: str, collection_name: str) -> pd.DataFrame:
    try:
        mongo_client = MongoClient("mongodb+srv://garvitgupta889:I0bvdn2v2EBwmLF5@garvit.hapddjc.mongodb.net/")

        collection = mongo_client[db_name][collection_name]

        # Fetch data from MongoDB collection
        data = list(collection.find())
        
        # Convert to DataFrame
        df = pd.DataFrame(data)

        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        df.replace({"na": np.nan}, inplace=True)

        return df

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X, y, models):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)




