import pickle
import uvicorn
from typing import Dict, Any 
from fastapi import FastAPI

app = FastAPI(title="churn-prediction", version="1.0.0")

with open ('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
    # X = dv.transform(customer)
    # churn = model.predict_proba(X)[0, 1]
    result = pipeline.predict_proba(customer)[0,1]
    return float(result)


@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)
    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)




