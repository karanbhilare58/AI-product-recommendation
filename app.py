from flask import Flask, render_template, jsonify, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load & Prepare Dataset ──────────────────────────────────────────
try:
    df = pd.read_csv("data.csv").drop_duplicates(subset=["product_id"])
    logger.info(f"✅ Loaded {len(df)} products across {df['category'].nunique()} categories")
except Exception as e:
    logger.error(f"❌ Failed to load data: {e}")
    df = pd.DataFrame()

# ─── Build Similarity Matrix ─────────────────────────────────────────
cosine_sim = []
if not df.empty:
    df["price_norm"] = df["price"] / df["price"].max()

    features = pd.concat([
        pd.get_dummies(df["category"]) * 5.0,   # category weight
        df[["price_norm"]]
    ], axis=1)

    cosine_sim = cosine_similarity(features, features)

# ─── Recommendation Engine ───────────────────────────────────────────
def recommend(product_id, top_n=4):
    """Return top_n similar products for a given product_id."""
    if df.empty:
        return {"error": "Dataset not loaded"}

    if product_id not in df["product_id"].values:
        return {"error": f"Product {product_id} not found"}

    idx = df[df["product_id"] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the product itself (index 0)
    top = sim_scores[1:top_n + 1]
    indices = [i[0] for i in top]
    scores  = [round(float(i[1]), 4) for i in top]

    result = df.iloc[indices][
        ["product_name", "category", "price", "image_url"]
    ].to_dict("records")

    # Attach similarity scores
    for rec, score in zip(result, scores):
        rec["similarity"] = score

    return {"recommendations": result}

# ─── Routes ───────────────────────────────────────────────────────────
@app.route("/")
def home():
    if df.empty:
        return "Error: dataset not loaded", 500

    catalog = df[
        ["product_id", "product_name", "category", "price", "image_url"]
    ].to_dict("records")

    categories = sorted(df["category"].unique().tolist())

    stats = {
        "total": len(df),
        "categories": df["category"].nunique()
    }

    return render_template("index.html", catalog=catalog, categories=categories, stats=stats)


@app.route("/recommend/<int:product_id>")
def get_recommendations(product_id):
    top_n = request.args.get("n", 4, type=int)
    top_n = min(max(top_n, 1), 10)  # clamp between 1-10
    return jsonify(recommend(product_id, top_n))


@app.route("/api/products")
def api_products():
    """Simple JSON endpoint listing all products."""
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500

    products = df[
        ["product_id", "product_name", "category", "price", "image_url"]
    ].to_dict("records")

    return jsonify({"products": products, "count": len(products)})


# ─── Run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)