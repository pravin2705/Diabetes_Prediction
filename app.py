from flask import Flask,jsonify,render_template,request
import numpy as np

import pickle

app = Flask(__name__)
model = pickle.load(open('log.pkl','rb'))


#Home Path
@app.route('/')
def Home():
    return render_template("home.html")

#Prediction
@app.route('/predict',methods =['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("home.html", prediction_text = "The person has diabetes {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)