# Synthetic Training Data

These files are synthetic examples created only to unblock local model export
and deployment smoke testing.

They are not the original ADEGuard training datasets, not VAERS-derived research
data, and not suitable for clinical validation or model performance claims.

Files:

- `project1.conll`: small BIO-labelled NER dataset for token classification.
- `dataset_with_entities_and_weaklabels.csv`: small severity classification
  dataset with weak labels.

Use these files to verify the MLOps path:

```text
training input -> model export -> preflight check -> Streamlit startup
```

Replace them with real approved training data before evaluating model quality or
making any research or clinical claims.
