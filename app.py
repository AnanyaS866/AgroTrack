from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/price_prediction_model.pkl")
df = pd.read_csv("data/cleaned_mandi_prices.csv")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        crop = request.form["crop"].strip()
        market = request.form["market"].strip()
        district = request.form["district"].strip()

        quantity = float(request.form["quantity"])
        year = int(request.form["year"])
        month = int(request.form["month"])
        day = int(request.form["day"])

        # Match Crop + Market + District
        matching = df[
            (df["Crop"].astype(str).str.strip().str.lower() == crop.lower()) &
            (df["Market"].astype(str).str.strip().str.lower() == market.lower()) &
            (df["District"].astype(str).str.strip().str.lower() == district.lower())
        ]

        # If exact combination is unavailable, use Crop
        if matching.empty:
            matching = df[
                df["Crop"].astype(str).str.strip().str.lower() == crop.lower()
            ]

        # If crop is also unavailable, use complete dataset
        if matching.empty:
            matching = df

        # Automatically obtain historical Min/Max
        min_price = matching["Min_Price"].mean()
        max_price = matching["Max_Price"].mean()

        input_data = pd.DataFrame({
            "Crop": [crop],
            "Market": [market],
            "District": [district],
            "Quantity": [quantity],
            "Min_Price": [min_price],
            "Max_Price": [max_price],
            "Year": [year],
            "Month": [month],
            "Day": [day]
        })

        prediction = model.predict(input_data)[0]

        # Keep prediction positive
        prediction = max(0, prediction)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)