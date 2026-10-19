# Synthetic Severity Evaluation Fixture

`severity_demo.jsonl` contains 12 invented cases, four per class. These are toy
labels for learning the evaluation pipeline, not medical severity guidance.
They describe reported sequences only; they establish no vaccine causality.

The illustrative labeling convention is minor/transient effects for Mild,
disrupted activities/outpatient care for Moderate, and admission/intensive care
or a fatal outcome for Severe. It is deliberately simplified, has no clinical
adjudication, and must not be used to label real cases. Every row explicitly
declares `synthetic: true`. The evaluator accepts only that declared scope.

The examples are separate from `data/training/` and are not consumed by training
scripts. The existing served weights are random synthetic smoke-test models.
Scores on these 12 cases teach reporting and failure analysis; they do not
measure generalization to real reports or validate the original research model.
For a real benchmark, obtain approved representative data, defensible labeling,
independent held-out splits, and a suitable trained model first.

From the project root with the Python environment active:

```powershell
python -m scripts.evaluate_severity
```

This verifies the severity artifact manifest, loads the shared serving model,
and writes a new timestamped JSON report under ignored `outputs/evaluation/`.
Optional `--dataset`, `--model`, and `--output` paths override the defaults;
existing output files are refused. Inference errors fail the run rather than
silently excluding difficult examples. No training or model changes occur.

Accuracy is the fraction of correct predictions. Precision for a class asks
how many predictions of that class were correct. Recall asks how many actual
examples of that class were found. F1 is the harmonic mean of precision and
recall; macro F1 averages the three class F1 values equally. All three labels
are always included; undefined ratios are reported as zero, including absent
classes. Support reports the number of actual examples per class. The confusion
matrix uses actual labels as rows and predicted labels as columns, in order
Severe, Moderate, Mild.

The report includes model version/manifest identity, exact dataset SHA-256,
runtime versions/device, and expected/predicted labels and confidence by case
ID. Raw input text is omitted from the report; look up IDs in the fixture.
Confidence is a model output, not a correctness or calibration guarantee.

A successful exit means the evaluation ran and produced a report. It does not
mean a model passed a quality gate. Establish a meaningful baseline and data
before selecting thresholds. Comparing results also requires controlling code
and runtime; the report is not a complete reproduction environment or a signed
provenance record. Retain the associated Git revision when comparing reports.
