import pandas as pd
from flask import Flask,render_template,request,redirect,url_for
import pickle
import numpy as np

app=Flask(__name__)
data=pd.read_csv('new_insurance_data.csv')
model=pickle.load(open("linearregression.pkl",'rb'))

@app.route('/')
def index():
    sex=sorted(data['sex'].unique())
    smokers = sorted(data['smoker'].unique())
    regions = sorted(data['region'].unique())
    return render_template('index.html',sex=sex,smokers=smokers,regions=regions)

@app.route('/predict',methods=["GET","POST"])
def predict():
    pred = None
    if request.method == 'POST':
        # Retrieve form values, ensure default values if fields are empty
        age = request.form.get('age', type=float)
        gender = request.form.get('gender')
        bmi = request.form.get('bmi', type=float)
        child = request.form.get('child', type=int) or 0
        Smoke = request.form.get('Smoke')
        camount = request.form.get('camount', type=float)
        pconsult = request.form.get('pconsult', type=int) or 0
        nsteps = request.form.get('nsteps', type=int) or 0
        hexpenditure = request.form.get('hexpenditure', type=float)
        phospitalizations = request.form.get('phospitalizations', type=int) or 0
        aSalary = request.form.get('aSalary', type=float)
        region = request.form.get('region')

        print(f"age: {age}, gender: {gender}, bmi: {bmi}, child: {child}, Smoke: {Smoke}, camount: {camount}, pconsult: {pconsult}, nsteps: {nsteps}, hexpenditure: {hexpenditure}, phospitalizations: {phospitalizations}, aSalary: {aSalary}, region: {region}")

        # Handle categorical encoding (as done before)
        gender_encoded = 0 if gender == 'male' else 1
        Smoke_encoded = 1 if Smoke == 'yes' else 0

        # Check if any of the numeric fields are missing or None
        if None in [age, bmi, child, camount, pconsult, nsteps, hexpenditure, phospitalizations, aSalary]:
            return render_template('index.html', pred="Error: Missing or invalid input values")

        # Create a DataFrame with the input values
        input_df = pd.DataFrame([[age, gender_encoded, bmi, child, Smoke_encoded, camount, pconsult, nsteps, hexpenditure, phospitalizations, aSalary, region]],
                                columns=['age', 'gender', 'bmi', 'child', 'Smoke', 'camount', 'pconsult', 'nsteps', 'hexpenditure', 'phospitalizations', 'aSalary', 'region'])

        # One-hot encoding if necessary (region encoding, if more than two categories)
        input_df = pd.get_dummies(input_df, columns=['region'])

        # Now call the actual model's predict method
        try:

            prediction = model.predict(input_df)
            print(prediction)
            prediction = model.predict(input_df)[0]
            pred=str(np.round(prediction,2))
        except ValueError as e:
            pred = f"Error in prediction: {e}"

    return render_template('index.html', pred=pred)


if __name__=="__main__":
    app.run(debug=True)