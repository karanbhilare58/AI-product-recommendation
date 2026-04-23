from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
try:
    df = pd.read_csv("data.csv").drop_duplicates(subset=["product_id"])
    print("✅ Data Loaded")
except Exception as e:
    print("❌ Error:", e)
    df = pd.DataFrame()

# Build model
if not df.empty:
    df["price_norm"] = df["price"] / df["price"].max()

    features = pd.concat([
        pd.get_dummies(df["category"]) * 5.0, # Give higher weight to category
        df[["price_norm"]]
    ], axis=1)

    cosine_sim = cosine_similarity(features, features)
else:
    cosine_sim = []

def recommend(product_id):
    if df.empty:
        return {"error": "Dataset not loaded"}

    if product_id not in df["product_id"].values:
        return {"error": "Invalid product ID"}

    idx = df[df["product_id"] == product_id].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top = sim_scores[1:4]
    indices = [i[0] for i in top]

    result = df.iloc[indices][
        ["product_name", "category", "price", "image_url"]
    ].to_dict("records")

    return {"recommendations": result}

@app.route("/")
def home():
    if df.empty:
        return "Error loading dataset"

    catalog = df[
        ["product_id", "product_name", "category", "price", "image_url"]
    ].to_dict("records")

    return render_template("index.html", catalog=catalog)

@app.route("/recommend/<int:product_id>")
def get_recommendations(product_id):
    return jsonify(recommend(product_id))

if __name__ == "__main__":
    app.run(debug=True)