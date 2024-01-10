from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# def pred_ml():
#     with open('/home/adithyagovish93/project/model.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     k = [[1.5666910077977176, 3.0, 4.0, 1.7588058573127272, 1, 3, 3.0, 0, 1, 3, 1, 0]]
#     prediction = loaded_model.predict(k)
#     return prediction[0]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        model = pickle.load(open('/home/adithyagovish93/project/mymodel2.pkl','rb'))

        # Get feature inputs from the form
        duration_of_pitch = float(request.form['duration_of_pitch'])
        number_of_followups = float(request.form['number_of_followups'])
        number_of_trips = float(request.form['number_of_trips'])

        monthly_income = float(request.form['monthly_income'])

        city_tier = int(request.form['city_tier'])
        product_pitched = int(request.form['product_pitched'])
        preferred_property_star = int(request.form['preferred_property_star'])
        passport = int(request.form['passport'])
        pitch_satisfaction_score = int(request.form['pitch_satisfaction_score'])
        designation = int(request.form['designation'])

        # marital_status_married = int(request.form['marital_status_married'])
        # marital_status_unmarried = int(request.form['marital_status_unmarried'])

        marital_status = int(request.form['marital_status'])

        if marital_status == 1:
            marital_status_married = 1
            marital_status_unmarried = 0
        else:
            marital_status_married = 0
            marital_status_unmarried = 1

        # Make a prediction using the trained model
        input_features = [duration_of_pitch, number_of_followups, number_of_trips, monthly_income, city_tier,
                                    product_pitched, preferred_property_star, passport, pitch_satisfaction_score,
                                    designation, marital_status_married, marital_status]
        # prediction = pred_ml()

        prediction = model.predict([input_features])
        if prediction == 1:
            outcome = 'Customer will buy the package'
        elif prediction == 0:
            outcome = 'Customer will not buy the package'

        return render_template('result.html', outcome=outcome)


        #result = model.predict(input_features)
        #return render_template('result.html', outcome=f'THE RESULT IS {prediction}')


if __name__ == '__main__':
    app.run(port=8080, debug=True)
