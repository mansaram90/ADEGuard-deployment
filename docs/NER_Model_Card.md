
# **ADEGuard BioBERT NER Model Card**

**Model Name:** ADEGuard-BioBERT-NER

**Version:** 1.0

**Authors / Maintainers:** Suganya G/CureviaAI

**Model Type:** Token classification (NER) based on BioBERT

---

## **Intended Use**

* **Primary Purpose:** Extract Adverse Drug Events (ADEs) and drug mentions from free-text symptom narratives in VAERS reports or similar clinical text.
* **Applications:**

  * Pharmacovigilance and drug safety monitoring.
  * Real-time ADE surveillance for hospitals, regulators, and pharmaceutical companies.
  * Input to downstream clustering, severity classification, and explainable dashboards (ADEGuard).
* **Input:** Free-text symptom descriptions.
* **Output:** Token-level ADE/DRUG predictions, post-processed entity spans.

---

## **Limitations**

* **Domain-specific:** Trained on VAERS vaccine reports (2020–2025); performance may degrade on non-vaccine ADEs or very different clinical text.
* **NER Errors:** May miss rare ADEs or drugs not in the training vocabulary.
* **Severity Not Included:** Model only extracts entities; severity classification is handled separately via Severity Classifier BioBERT model.
* **No Causality Detection:** The model identifies mentions but cannot confirm causal relationship between drug and ADE.
* **Post-processing Reliance:** Fuzzy matching and dictionary-based normalization improve accuracy but may introduce false positives.
* **Long Text Handling:** Input truncated to 512 tokens due to BioBERT limits; very long narratives may lose context.

---

## **Performance**

* **Entity-level accuracy:** Post-processing improved matching to known drug/ADE dictionaries.
* **Clustering & downstream tasks:** Embeddings + clustering revealed age- and modifier-aware symptom patterns.

> ⚠️ Performance varies with domain, text quality, and presence of unseen drugs/ADEs.

---

## **Ethical Considerations**

* **Intended for professional use** by clinicians, regulators, or pharmacovigilance teams.
* **Not a diagnostic tool:** Outputs should be interpreted by experts; misinterpretation may lead to inappropriate decisions.
* **Data Privacy:** Model should be used on de-identified clinical reports or VAERS data in compliance with regulations.

---

## **Training Data**

* VAERS vaccine reports (2020–2025) with gold-annotated ADE/DRUG spans.
* Augmented with weak labels from structured symptom fields.

---

## **Metrics**
### **NER Training Summary**

| Step | Training Loss | Validation Loss | Precision | Recall | F1 Score |
| ---- | ------------- | --------------- | --------- | ------ | -------- |
| 500  | 0.0652        | 0.0287          | 75.43%    | 98.93% | 85.60%   |
| 1000 | 0.0327        | 0.0159          | 86.38%    | 99.52% | 92.48%   |
| 1500 | 0.0143        | 0.0115          | 91.99%    | 99.56% | 95.63%   |
| 2000 | 0.0114        | 0.0114          | 89.89%    | 99.81% | 94.59%   |
| 2500 | 0.0072        | 0.0099          | 92.66%    | 99.73% | 96.07%   |
| 3000 | 0.0182        | 0.0099          | 95.48%    | 99.78% | 97.58%   |
| 3500 | 0.0063        | 0.0076          | 94.81%    | 99.75% | 97.22%   |
| 4000 | 0.0045        | 0.0099          | 96.15%    | 99.82% | 97.95%   |
| 4500 | 0.0053        | 0.0088          | 96.46%    | 99.81% | 98.11%   |

**Test Set Performance:**

* Loss: 0.00564
* Precision: 96.27%
* Recall: 99.75%
* F1 Score: 97.98%

**Interpretation:**

* Training and validation loss steadily decrease → good learning.
* Precision and F1 increase over steps → the model correctly identifies ADE/Drug entities.
* High recall indicates very few ADE/Drug entities are missed.

