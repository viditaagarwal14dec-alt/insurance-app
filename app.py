from flask import Flask, request, render_template_string
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("insurance_expenses_model.pkl")

# Flask app
app = Flask(__name__)

# ---------- HTML TEMPLATE WITH PROFESSIONAL UI ----------
FORM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Insurance Expense Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <!-- Google Font -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            background: radial-gradient(circle at top left, #4f46e5, #111827);
            color: #111827;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        .shell {
            max-width: 1100px;
            width: 100%;
        }
        .card {
            background: #f9fafb;
            border-radius: 18px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.35);
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 3fr) minmax(0, 2.4fr);
        }
        @media (max-width: 900px) {
            .card {
                grid-template-columns: 1fr;
            }
        }
        .card-left {
            padding: 32px 32px 28px;
        }
        .card-right {
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            color: #e5e7eb;
            padding: 32px 28px 28px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            background: #eef2ff;
            color: #4f46e5;
            font-weight: 600;
            margin-bottom: 10px;
        }
        h1 {
            margin: 0 0 6px;
            font-size: 26px;
            font-weight: 600;
            color: #111827;
        }
        .subtitle {
            margin: 0 0 22px;
            font-size: 13px;
            color: #4b5563;
        }
        form {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px 18px;
        }
        @media (max-width: 640px) {
            form {
                grid-template-columns: 1fr;
            }
        }
        .field {
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 13px;
        }
        label {
            font-weight: 500;
            color: #374151;
        }
        input[type="number"],
        select {
            border-radius: 10px;
            border: 1px solid #d1d5db;
            padding: 8px 10px;
            font-size: 13px;
            outline: none;
            transition: border-color 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease;
            background-color: #ffffff;
        }
        input[type="number"]:focus,
        select:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.15);
        }
        .hint {
            font-size: 11px;
            color: #9ca3af;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        button {
            margin-top: 6px;
            border: none;
            border-radius: 999px;
            padding: 9px 18px;
            font-size: 13px;
            font-weight: 500;
            background: linear-gradient(135deg, #4f46e5, #6366f1);
            color: white;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.15s ease;
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.35);
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(79, 70, 229, 0.45);
        }
        button:active {
            transform: translateY(0);
            box-shadow: 0 7px 16px rgba(79, 70, 229, 0.35);
        }
        .footer-note {
            margin-top: 14px;
            font-size: 11px;
            color: #9ca3af;
        }

        /* Right side (result panel) */
        .result-title {
            font-size: 18px;
            margin: 0 0 8px;
            font-weight: 600;
        }
        .result-sub {
            margin: 0 0 30px;
            font-size: 13px;
            color: #d1d5db;
        }
        .empty-state {
            font-size: 13px;
            line-height: 1.6;
            color: #d1d5db;
        }
        .amount-card {
            background: rgba(15, 23, 42, 0.65);
            border-radius: 14px;
            padding: 18px 16px 16px;
            margin-bottom: 16px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.55);
        }
        .amount-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #9ca3af;
            margin-bottom: 4px;
        }
        .amount-value {
            font-size: 26px;
            font-weight: 600;
            color: #f9fafb;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .pill {
            font-size: 11px;
            padding: 4px 9px;
            border-radius: 999px;
            background-color: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.4);
            color: #e5e7eb;
        }
        .corner-tag {
            font-size: 11px;
            text-align: right;
            color: #c7d2fe;
        }
    </style>
</head>
<body>
<div class="shell">
    <div class="card">
        <!-- LEFT: FORM -->
        <div class="card-left">
            <div class="badge">ML powered • Instant quote</div>
            <h1>Medical Insurance Expense Predictor</h1>
            <p class="subtitle">
                Enter a few details about the person and get an estimated annual insurance expense.
            </p>

            <form action="/predict" method="post">
                <div class="field">
                    <label for="age">Age</label>
                    <input type="number" name="age" id="age" min="0" max="120" required placeholder="e.g., 32" />
                </div>

                <div class="field">
                    <label for="sex">Sex</label>
                    <select name="sex" id="sex">
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                    </select>
                </div>

                <div class="field">
                    <label for="bmi">BMI</label>
                    <input type="number" step="0.1" name="bmi" id="bmi" min="10" max="60"
                           required placeholder="e.g., 27.5" />
                    <span class="hint">Body Mass Index</span>
                </div>

                <div class="field">
                    <label for="children">Number of children</label>
                    <input type="number" name="children" id="children" min="0" max="10" required placeholder="0" />
                </div>

                <div class="field">
                    <label for="smoker">Smoker</label>
                    <select name="smoker" id="smoker">
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                    </select>
                </div>

                <div class="field">
                    <label for="region">Region</label>
                    <select name="region" id="region">
                        <option value="northeast">Northeast</option>
                        <option value="northwest">Northwest</option>
                        <option value="southeast">Southeast</option>
                        <option value="southwest">Southwest</option>
                    </select>
                </div>

                <div class="field full-width">
                    <button type="submit">
                        Predict Expense
                        <span>➜</span>
                    </button>
                    <p class="footer-note">
                        This is a demo tool built for learning purposes. Estimates are based on a trained ML model
                        and do not replace an official insurance quote.
                    </p>
                </div>
            </form>
        </div>

        <!-- RIGHT: RESULT / INFO -->
        <div class="card-right">
            <div>
                <p class="corner-tag">Insurance Analytics • Demo</p>
                <h2 class="result-title">Your estimated annual expense</h2>
                <p class="result-sub">
                    Powered by a machine learning model trained on real insurance data.
                </p>

                {% if prediction is not none %}
                <div class="amount-card">
                    <div class="amount-label">Predicted yearly premium</div>
                    <div class="amount-value">
                        ${{ '{:,.2f}'.format(prediction) }}
                    </div>
                    <div class="pill-row">
                        <span class="pill">Personalised estimate</span>
                        <span class="pill">Based on your inputs</span>
                    </div>
                </div>
                {% else %}
                <p class="empty-state">
                    Fill out the form on the left and click <strong>“Predict Expense”</strong>.
                    Your estimated annual medical insurance cost will appear here.
                </p>
                {% endif %}
            </div>

            <div class="empty-state" style="font-size: 11px; margin-top: 22px;">
                Built with Flask, scikit-learn & Render • For academic / portfolio use.
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""

# ---------- ROUTES ----------

@app.route("/", methods=["GET"])
def home():
    # By default, no prediction yet
    return render_template_string(FORM_HTML, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        # Build a DataFrame exactly like training
        df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }])

        pred = model.predict(df)[0]
        pred = round(float(pred), 2)

        return render_template_string(FORM_HTML, prediction=pred)

    except Exception as e:
        # In production you would log this
        return render_template_string(FORM_HTML, prediction=None)


if __name__ == "__main__":
    # Local run
    app.run(host="0.0.0.0", port=5000, debug=True)
