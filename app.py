from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)
model = pickle.load(open("breast_cancer_detecter.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods =["POST"])
def predict():
    input_names =[float(x) for x in request.form.values()]
    feature_val =[np.array(input_names)]

    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
    df = pd.DataFrame(feature_val,columns=features_name)
    output = model.predict(df)
    if output ==0:
        res_val = "patient has breast cancer!😪"
    else:
        res_val = "patient has no breast Cancer🥰"

        return render_template("index.html",prediction_text="patient  has{}".format(res_val))
if __name__=="__main__":
    app.run()