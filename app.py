import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    dict1 = {'age': 0, "Number of Sexual Partners":0, "Age when has First Sexual intercourse":0, "Number of Pregnancies":0, "smoke":0, "Smoke (Years)":0, "Smoke (Packs/Year)":0, "Hormonal":0, "Hormonal Contraceptives (years)":0, "IUD":0, "IUD (years)":0, "STD":0, "STD (number)":0, "item1":0, "item2":0, "item3":0, "item4":0, "item5":0, "item6":0, "item7":0, "item8":0, "item9":0, "item10":0, "item11":0, "item12":0, "Number of diagnosis":0, "cancer":0, "std1":0, "std2":0, "std3":0, "std4":0, "hinselmann":0, "schiller":0, "cytology":0 }
    dict2 = request.form.to_dict()
    for key, value in dict1.items():
        if dict2.get(key):
            dict1[key] = dict2[key]
    int_features = [int(float(x)) for x in dict1.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])
    if (output == 1):
        text = 'You have risk of Cervical Cancer'
    if (output == 0):
        text = "You haven't any risk of Cervical Cancer"

    return render_template('index.html', prediction_text=text)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
