import pickle
from flask import Flask
from Flask import request
from Flask import j
model file = 'model_C=1.0.bin'

with open(model_fime, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app =Flask('churn')

@app.route('/predict', methods=['POST'])

def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability ': float(y_pred),
        'churn': churn
    }
    return result

    print('input:', customer)
    print('output:', y_pred)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


