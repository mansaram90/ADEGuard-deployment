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

## Reproducible severity preparation

Before candidate training, run from the project root with the Python environment
active:

```powershell
python -m scripts.prepare_severity_data
```

The command reads this synthetic severity CSV, validates text/labels, assigns
stable IDs, and writes a NEW timestamped package beneath ignored
`outputs/datasets/severity/`. It does not train or modify active model artifacts.
Optional `--source`, `--evaluation`, `--seed`, and `--output-dir` select another
declared synthetic CSV, protected synthetic evaluation fixture, integer seed,
or new output directory. Existing output directories are refused.

The CSV contract requires `symptom_combined`, `weak_label_id`, and `weak_label`.
Label mappings must agree: 0=Severe, 1=Moderate, 2=Mild. Blank texts, malformed
rows/headers, unknown or inconsistent labels, duplicate normalized texts,
and conflicts on the same text fail preparation. All three classes require
at least three examples so each split can contain one.

IDs are `syn-sev-` plus SHA-256 of casefolded, whitespace-collapsed text.
They are stable under CSV reordering, case/whitespace-only differences, and
label correction. Raw model text is retained in output; ID normalization does
not alter the text used for training. The IDs identify synthetic text cases,
not real-world patients/reports, and are not an anonymization guarantee.

Within each class, SHA-256 of seed plus case ID ranks examples. Seed 42 is
the default. The algorithm targets 70/15/15 percent, uses integer largest-
remainder rounding with train/validation/test tie order, and ensures at least
one example of each class in each split. This deterministic hash ranking
does not rely on input row order or a random-number library implementation.
Small counts mean actual percentages differ from the requested ratios.

For the current 45-row source:

| Split | Severe | Moderate | Mild | Total |
|---|---:|---:|---:|---:|
| Train | 7 | 16 | 9 | 32 |
| Validation | 2 | 3 | 2 | 7 |
| Test | 1 | 3 | 2 | 6 |

Use train to update model weights, validation to select configurations or
checkpoints, and test for a final assessment after selection. Reuse the same
saved package across candidate comparisons. Changing source data, labels, seed
or algorithm can change assignments; stable IDs do not promise fixed splits
after those changes. The six-case test is a teaching artifact with highly
uncertain scores, not a real benchmark.

The separate 12-case `evaluation/severity_demo.jsonl` stays outside all splits.
Preparation rejects exact normalized text overlap with it, rejects overlap
between prepared rows, and checks every source ID appears exactly once.
These checks do not detect semantic near-duplicates or grouped/patient/time
leakage needed for a real dataset.

Each package has `train.jsonl`, `validation.jsonl`, `test.jsonl`, and
`manifest.json`. The manifest records exact source/evaluation byte fingerprints,
algorithm versions, seed, requested ratios, actual counts/class counts/IDs,
and SHA-256 of the output files. It is written last; a package without a valid
manifest is incomplete and must not be used. Preparation refuses overwrite
and does not clean up a partial directory after an I/O failure. A future
training consumer must verify the manifest and output file fingerprints before
using a saved package.

The source CSV's Git attributes preserve exact bytes on Windows/Linux. Hashes
identify the content, not trusted authorship or label correctness. The manifest
does not include timestamps/output locations, so identical inputs and seed
produce byte-identical package files even in different output directories.
