from flask import Flask, request, render_template_string
import pandas as pd
import joblib

# Load the model file
model = joblib.load("insurance_expenses_model.pkl")

app = Flask(__name__)

# This is the HTML of your webpage
HTML = """
<h2>Insurance Expense Predictor</h2>
<form method="POST" action="/predict">
  Age: <input name="age" type="number" required><br><br>
  Sex: 
  <select name="sex">
    <option value="male">Male</option>
    <option value="female">Female</option>
  </select><br><br>
  BMI: <input name="bmi" type="number" step="0.1" required><br><br>
  Children: <input name="children" type="number" required><br><br>
  Smoker: 
  <select name="smoker">
    <option value="yes">Yes</option>
    <option value="no">No</option>
  </select><br><br>
  Region:
  <select name="region">
    <option value="northeast">Northeast</option>
    <option value="northwest">Northwest</option>
    <option value="southeast">Southeast</option>
    <option value="southwest">Southwest</option>
  </select><br><br>

  <button type="submit">Predict</button>
</form>

{% if result is not none %}
  <h3>Predicted Expense: {{ result }}</h3>
{% endif %}
"""

@app.route("/")
def home():
    return render_template_string(HTML, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "age": int(request.form["age"]),
        "sex": request.form["sex"],
        "bmi": float(request.form["bmi"]),
        "children": int(request.form["children"]),
        "smoker": request.form["smoker"],
        "region": request.form["region"]
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    prediction = round(float(prediction), 2)

    return render_template_string(HTML, result=prediction)

# No app.run() here (Render runs it automatically)
