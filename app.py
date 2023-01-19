from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('models/cardiomod.h5')
scaler = joblib.load('models/scalemod.h5')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    data = request.args
    age = int(data['age'])
    ap_hi = int(data['ap_hi'])
    ap_lo = int(data['ap_lo'])
    bmi = float(data['bmi'])
    weight = float(data['weight'])
    hig = float(data['hig'])
    blood = int(data['blood pressure'])
    glu = int(data['gluc'])
    alco = int(data['alco'])
    gender = int(data['gen'])
    smoke = int(data['smoke'])
    active = int(data['ac'])
    cho = int(data['cho'])

    dt = [gender, hig, weight, ap_hi, ap_lo, cho, glu, smoke, alco, active, age, bmi, blood]
    dt = scaler.transform([dt])
    pred = model.predict(dt)
    def fun(p):
        if p==0:
            return 'you are okay no problem , Do not skip breakfast Eat regular meals Eating at regular times during the day helps burn calories at a faster rate. It also reduces the temptation to snack on foods high in fat and sugar. Eat plenty of fruit and veg'
        else:
            return 'try to keep your health you could have cardio vascular'

    return render_template('prediction.html', use=fun(pred))


if __name__ == '__main__':
    app.run()
