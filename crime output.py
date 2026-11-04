# crime_rate_ml_project.py
# County-level violent crime rate prediction (Maryland Open Data)
# Run in IDLE: open this file -> Run -> Run Module (F5)
# The CSV must be in the same folder and named crime.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE = "crime.csv"
TARGET = "VIOLENT CRIME RATE PER 100,000 PEOPLE"
PROP = "PROPERTY CRIME RATE PER 100,000 PEOPLE"

# ---------- 1. Load and explore ----------
df = pd.read_csv(FILE)
df.columns = [c.strip().upper() for c in df.columns]
print("Shape:", df.shape)
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
print(df[["POPULATION", TARGET, PROP]].describe())

# ---------- 2. Preprocessing ----------
df = df.drop_duplicates()
df = df.sort_values(["JURISDICTION", "YEAR"]).reset_index(drop=True)
df["PREV_VIOLENT_RATE"] = df.groupby("JURISDICTION")[TARGET].shift(1)
df["PREV_PROPERTY_RATE"] = df.groupby("JURISDICTION")[PROP].shift(1)
df = df.dropna(subset=["PREV_VIOLENT_RATE", "PREV_PROPERTY_RATE", TARGET])
print("Usable records:", len(df))

NUM = ["POPULATION", "PREV_VIOLENT_RATE", "PREV_PROPERTY_RATE"]
CAT = ["JURISDICTION"]

# Correlation of numeric inputs with the target
corr = df[NUM + [TARGET]].corr()
print("Correlation matrix:")
print(corr.round(3))

# Chronological split: before 2010 = train, 2010 onwards = test
train = df[df["YEAR"] < 2010]
test = df[df["YEAR"] >= 2010]
print("Train:", len(train), " Test:", len(test))
X_train, y_train = train[NUM + CAT], train[TARGET]
X_test, y_test = test[NUM + CAT], test[TARGET]


def make_pre(poly=False, scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if poly:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    if scale:
        steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


# ---------- 3. Models ----------
models = {
    "Multiple Linear Regression": Pipeline([
        ("pre", make_pre()), ("m", LinearRegression())]),
    "Polynomial Regression (deg 2)": Pipeline([
        ("pre", make_pre(poly=True)), ("m", LinearRegression())]),
    "Decision Tree": Pipeline([
        ("pre", make_pre(scale=False)),
        ("m", DecisionTreeRegressor(max_depth=5, random_state=42))]),
    "Random Forest": Pipeline([
        ("pre", make_pre(scale=False)),
        ("m", RandomForestRegressor(n_estimators=200, random_state=42))]),
    "SVR (RBF kernel)": Pipeline([
        ("pre", make_pre()),
        ("m", TransformedTargetRegressor(
            regressor=SVR(kernel="rbf", C=10, epsilon=0.05),
            transformer=StandardScaler()))]),
}


def scores(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"Model": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred)}


# ---------- 4. Train and evaluate ----------
rows = []
preds = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    rows.append(scores(name, y_test, p))

# Simple baseline: predict this year's rate = last year's rate
rows.append(scores("Baseline (previous-year rate)", y_test, test["PREV_VIOLENT_RATE"]))

results = pd.DataFrame(rows).round(4)
print(results.to_string(index=False))
results.to_csv("results.csv", index=False)

# ---------- 5. Figures ----------
model_rows = results.iloc[:5]
best = model_rows.sort_values("RMSE").iloc[0]["Model"]
print("Best model by RMSE:", best)

# Fig A: actual vs predicted for the best model
plt.figure(figsize=(5, 5))
plt.scatter(y_test, preds[best], s=12, alpha=0.6)
lo, hi = y_test.min(), y_test.max()
plt.plot([lo, hi], [lo, hi], "r--")
plt.xlabel("Actual violent crime rate")
plt.ylabel("Predicted violent crime rate")
plt.title("Actual vs predicted: " + best)
plt.tight_layout()
plt.savefig("fig_actual_vs_predicted.png", dpi=200)
plt.close()

# Fig B: RMSE of all models
plt.figure(figsize=(7, 4))
plt.barh(results["Model"], results["RMSE"], color="steelblue")
plt.xlabel("RMSE (lower is better)")
plt.title("RMSE of the models on the test set")
plt.tight_layout()
plt.savefig("fig_rmse_comparison.png", dpi=200)
plt.close()

# Fig C: correlation heatmap
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
labels = ["Population", "Prev violent", "Prev property", "Violent rate"]
plt.xticks(range(4), labels, rotation=30, ha="right")
plt.yticks(range(4), labels)
for i in range(4):
    for j in range(4):
        plt.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center")
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("fig_correlation.png", dpi=200)
plt.close()

# Fig D: average violent crime rate per year (data exploration)
yearly = df.groupby("YEAR")[TARGET].mean()
plt.figure(figsize=(7, 4))
plt.plot(yearly.index, yearly.values)
plt.axvline(2010, color="red", linestyle="--", label="train/test split")
plt.xlabel("Year")
plt.ylabel("Mean violent crime rate")
plt.title("Average violent crime rate across counties")
plt.legend()
plt.tight_layout()
plt.savefig("fig_trend.png", dpi=200)
plt.close()

# Fig E: actual vs predicted for every model (one panel each)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
panels = list(preds.items()) + [("Baseline (previous-year rate)", test["PREV_VIOLENT_RATE"].values)]
for ax, (name, p) in zip(axes.ravel(), panels):
    ax.scatter(y_test, p, s=8, alpha=0.6)
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
plt.tight_layout()
plt.savefig("fig_all_models.png", dpi=200)
plt.close()

# Fig F: top levels of the decision tree
tree_pipe = models["Decision Tree"]
names = tree_pipe.named_steps["pre"].get_feature_names_out()
plt.figure(figsize=(12, 5))
plot_tree(tree_pipe.named_steps["m"], feature_names=names, max_depth=2,
          filled=True, fontsize=8)
plt.title("Decision tree (top levels)")
plt.tight_layout()
plt.savefig("fig_decision_tree.png", dpi=200)
plt.close()

# Fig G: most important inputs of the random forest
rf = models["Random Forest"]
imp = pd.Series(rf.named_steps["m"].feature_importances_,
                index=rf.named_steps["pre"].get_feature_names_out())
imp = imp.sort_values().tail(6)
plt.figure(figsize=(7, 4))
plt.barh([n.split("__")[-1] for n in imp.index], imp.values, color="seagreen")
plt.xlabel("Importance")
plt.title("Random forest: most important inputs")
plt.tight_layout()
plt.savefig("fig_rf_importance.png", dpi=200)
plt.close()
print(imp.round(4).to_string())

print("Done. Figures and results.csv saved in the current folder.")
import os
for f in ["fig_trend.png", "fig_correlation.png", "fig_rmse_comparison.png",
          "fig_actual_vs_predicted.png", "fig_all_models.png",
          "fig_decision_tree.png", "fig_rf_importance.png"]:
    os.startfile(f)
