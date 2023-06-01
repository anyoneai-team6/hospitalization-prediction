import json
import time
from uuid import uuid4

import redis
import settings

db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)


def model_predict(data_input):
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.
    
    Parameters
    ----------
    data_input : str
        Name for the image uploaded by the user.
    
    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    prediction = None
    score = None
    
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "data_input": data_input
    }
    
    res = db.lpush(settings.REDIS_QUEUE, json.dumps(job_data))
    
    while True:
        output = db.get(job_id)
        if output:
            results = json.loads(output.decode("utf-8"))
            prediction = results["prediction"]
            score = results["score"]
            
            db.delete(job_id)
            break
        time.sleep(settings.API_SLEEP)
    
    return prediction, score