from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
model_data = joblib.load("recession_model.pkl")

model = model_data["model"]
model_columns = model_data["columns"]
threshold = model_data["threshold"]

# Extract country list from model columns
countries = [col.replace("Country_", "") for col in model_columns if "Country_" in col]


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    prob_no = None
    prob_yes = None

    if request.method == "POST":

        country = request.form["country"]

        Lag_GDP = float(request.form["Lag_GDP"])
        Lag_GDP_2 = float(request.form["Lag_GDP_2"])
        Lag_GDP_3 = float(request.form["Lag_GDP_3"])

        Inflation = float(request.form["Inflation"])
        Unemployment = float(request.form["Unemployment"])
        Debt = float(request.form["Debt"])

        # Feature Engineering
        GDP_rolling_3 = (Lag_GDP + Lag_GDP_2 + Lag_GDP_3) / 3

        GDP_Variance = (
            (Lag_GDP - GDP_rolling_3) ** 2 +
            (Lag_GDP_2 - GDP_rolling_3) ** 2 +
            (Lag_GDP_3 - GDP_rolling_3) ** 2
        ) / 3

        GDP_volatility = np.sqrt(GDP_Variance)

        input_data = {
            "Lag_GDP": Lag_GDP,
            "Lag_GDP_2": Lag_GDP_2,
            "GDP_rolling_3": GDP_rolling_3,
            "GDP_volatility": GDP_volatility,
            "Inflation": Inflation,
            "Unemployment": Unemployment,
            "Debt": Debt
        }

        # Create dataframe with correct columns
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

        # Fill numerical features
        for feature in input_data:
            if feature in input_df.columns:
                input_df.at[0, feature] = input_data[feature]

        # Activate country dummy variable
        country_column = "Country_" + country

        if country_column in input_df.columns:
            input_df.at[0, country_column] = 1

        # Prediction
        probability = model.predict_proba(input_df)[0][1]

        prediction = 1 if probability > threshold else 0

        if prediction == 0:
            result = "Recession Likely"
        else:
            result = "No Recession"

        prob_no = round(probability * 100, 2)
        prob_yes = round((1 - probability) * 100, 2)

    return render_template(
        "index.html",
        result=result,
        prob_no=prob_no,
        prob_yes=prob_yes,
        countries=countries
    )


if __name__ == "__main__":
    app.run(debug=True)