import json
import time
import settings

import numpy as np
import xgboost as xgb


db = redis.Redis(
    host=settings.REDIS_IP ,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)

modelo = xgb.Booster()
modelo.load_model('xgb-model/modelo_xgb.xgb')


def predict(data):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    
    Parameters
    ----------
    data : str
        Image filename.
    
    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    pred = np.round(pred_probability)
    pred_probability = modelo.predict_proba(np.array([data]))[:,1]
    return pred, pred_probability


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
        class_name, pred_probability = predict(job["data"])
        
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