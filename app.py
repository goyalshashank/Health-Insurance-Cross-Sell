import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import LabelEncoder, RobustScaler

#Initialising Flask
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("Home.html")

#Predict Function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,10)
    df = pd.DataFrame(to_predict)
    
    scaler = pickle.load(open('scaler.pkl','rb'))
    
    scaler.transform(df)
    
    model = pickle.load(open('cross_sell_model.pkl','rb'))
                        
    result = model.predict(df)
    return result

#Output Page and Login
@app.route("/result",methods=['POST'])
def result():
    if request.method == 'POST':
        print("POST")
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = "Potential Customer for Cross Sell"
        else:
            prediction = "Not an Ideal person for cross sell"
        return render_template("result.html",prediction=prediction)
    
#Main Function
if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
        