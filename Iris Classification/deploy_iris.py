from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
# load the model
model = joblib.load(open('saved_model.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width  = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width  = float(request.form['petal_width'])

        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    except Exception as e:
        prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)