import pandas as pd
import re
import uuid
import json
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LabelingFunction, PandasLFApplier

from sklearn.model_selection import train_test_split

# -----------------------------
# Snorkel Labeling Functions
# -----------------------------

SEVERE, MODERATE, MILD, ABSTAIN = 0, 1, 2, -1
id2label = {0: "Severe", 1: "Moderate", 2: "Mild"}

@labeling_function()
def lf_died(row):
    return SEVERE if str(row.get("DIED","")).strip().upper() == "Y" else ABSTAIN

@labeling_function()
def lf_hospital(row):
    return SEVERE if str(row.get("HOSPITAL",0)) in ["1","Y","YES"] else ABSTAIN

@labeling_function()
def lf_l_threat(row):
    return SEVERE if str(row.get("L_THREAT","")).strip().upper() == "Y" else ABSTAIN

@labeling_function()
def lf_disable(row):
    return SEVERE if str(row.get("DISABLE","")).strip().upper() == "Y" else ABSTAIN

@labeling_function()
def lf_text_severe(row):
    text = str(row.get("SYMPTOM_TEXT","")).lower()
    return SEVERE if any(w in text for w in ["death","fatal","critical","severe","life threatening"]) else ABSTAIN

@labeling_function()
def lf_text_moderate(row):
    return MODERATE if "moderate" in str(row.get("SYMPTOM_TEXT","")).lower() else ABSTAIN

@labeling_function()
def lf_text_mild(row):
    return MILD if "mild" in str(row.get("SYMPTOM_TEXT","")).lower() else ABSTAIN

lfs = [lf_died, lf_hospital, lf_l_threat, lf_disable, lf_text_severe, lf_text_moderate, lf_text_mild]

# -----------------------------
# 5️⃣ Apply LFs & Train Label Model
# -----------------------------
df = pd.read_csv("ade_gold_subset_20k.csv")

applier = PandasLFApplier(lfs)
L_train = applier.apply(df)

label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)

# Probabilities + hard labels
probs = label_model.predict_proba(L=L_train)
df["weak_label_prob_SEVERE"] = probs[:, SEVERE]
df["weak_label_prob_MODERATE"] = probs[:, MODERATE]
df["weak_label_prob_MILD"] = probs[:, MILD]
df["weak_label_id"] = label_model.predict(L=L_train)
df["weak_label"] = df["weak_label_id"].map(id2label)

# Save outputs
df.to_csv("dataset_with_weaklabels.csv", index=False)

print("✅ ADE/DRUG extraction + Snorkel weak labeling completed!")

