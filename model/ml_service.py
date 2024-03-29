import json
import time
import settings
import pandas as pd
import numpy as np
import xgboost as xgb
import redis
import joblib

db = redis.Redis(
    host=settings.REDIS_IP ,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)

transformer = joblib.load('test/transformer.pkl')
scaler = joblib.load('test/scaler.pkl')
model = joblib.load('test/model2.pkl') 


def predict(data):
    """
    Predicts the probability and classification of a given dataset.

    Parameters:
    - data (dict): A dictionary containing input data as key-value pairs, where the keys represent the features/columns and the values represent the corresponding values.

    Returns:
    - pred (float): The rounded prediction/classification (0 or 1).
    - pred_probability (float): The predicted probability as a percentage.
    """
    
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.transpose().sort_index(axis=1)
    
    df['paheight']=df['paheight'] / 100
    df = scaler.transform(df)
    df = transformer.transform(df)
    
    pred_probability = model.predict_proba(df)[:, 1][0]
    pred = np.round(pred_probability)
    return float(pred), round(float(pred_probability*100),2)


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        job = db.brpop(settings.REDIS_QUEUE)[1]
        
        job = json.loads(job.decode("utf-8"))
        class_name, pred_probability = predict(job["data_input"])
        
        pred = {
            "prediction": class_name,
            "score": float(pred_probability),
        }
        
        job_id = job["id"]
        db.set(job_id, json.dumps(pred))
        
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    print("Launching ML service...")
    classify_process()