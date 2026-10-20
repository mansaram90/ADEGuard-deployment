# Synthetic Severity Candidate Training

This local exercise trains the tiny random severity model on the saved synthetic
data splits. It teaches candidate training, validation selection, lineage and
artifact export. It does not reproduce the original BioBERT research results,
train on real reports, establish clinical quality, or promote a serving model.

## Run from a verified data package

Prepare the data first using `python -m scripts.prepare_severity_data`. Use the
printed package path, keeping the same saved splits across candidate comparisons.
For the package created during this learning session:

```powershell
python -m scripts.train_severity_candidate --dataset outputs/datasets/severity/20261020T094655271040Z --version demo-severity-trained-v1
```

The CLI defaults to five epochs, batch size 8, learning rate 0.0005, seed 42,
and the configured local severity model as parent. Optional `--epochs`,
`--batch-size`, `--learning-rate`, `--seed`, `--parent`, `--evaluation`, and
`--output-dir` make settings explicit. Use a new artifact version/directory
for a new experiment. The default output is timestamped under ignored
`outputs/candidates/`. Existing output paths and output inside inputs or active
model directories are refused. No new training dependency or model download
is required: weights/tokenizer are loaded with `local_files_only=True`.

## What happens

1. `load_prepared_dataset()` verifies the manifest schema/synthetic scope,
   regular-file package layout, file hashes, row schemas, stable IDs, class
   counts, full coverage, protected evaluation identity/overlap, and exact
   membership implied by the recorded splitting algorithm and seed.
2. Parent model artifact hashes and sequence-classification task are verified.
   This trainer accepts only synthetic demo/candidate artifact types and the
   serving severity label mapping: 0 Severe, 1 Moderate, 2 Mild.
3. The local model and tokenizer load on CPU. Input lengths are checked without
   truncation using the same serving limit helper. Oversized train/validation
   examples fail rather than changing the evaluation population.
4. Parent validation metrics are recorded before training. Only the 32 train
   cases update weights. AdamW uses the configured learning rate, weight decay
   0.01 and gradient-norm clipping at 1.0. Batches are shuffled with a seeded
   torch generator. Nonfinite loss fails the run.
5. After each epoch, the model switches to evaluation mode and scores the seven
   validation cases through the shared `SeverityModel.predict()` method. The
   training loss and full validation metrics are retained in the report.
6. The highest validation macro F1 among trained epochs determines the exported
   checkpoint. Strict improvement replaces the retained in-memory weight copy;
   ties retain the earliest epoch. The parent is recorded for comparison, not
   silently substituted as the trained candidate if trained scores are worse.
7. The selected weights/tokenizer/labels, synthetic-trained metadata and training
   report are written under `<run>/model/`, then an artifact manifest is written
   last and verified. It covers the provenance report as well as model files.

The six test rows are read for package integrity only. They are never tokenized,
predicted, optimized against or used for checkpoint selection in this command.
The separate 12-case evaluation fixture is checked for identity/overlap, not
scored. Both report flags explicitly state those evaluations were not performed.
An assessment of the fixed selected candidate is a later step; repeatedly using
test results to select experiments would change its role.

CPU training uses one torch thread, an isolated seeded CPU RNG context, and
deterministic algorithms. Previous CPU RNG/thread/determinism settings are
restored after this offline call. Same-environment tests repeat training and
compare saved weight hashes. Bitwise equivalence across library releases,
platforms and hardware is not guaranteed; runtime and implementation hashes
are recorded for comparison. See [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## What to inspect

The output model folder contains:

- `model.safetensors`, `config.json`, tokenizer files and `label.txt`.
- `ADEGUARD_DEMO_ARTIFACT.json`, identifying a synthetic trained learning candidate.
- `training-report.json`, with parent identity, dataset manifest/source/split
  hashes, settings, runtime/code hashes, baseline, epoch history and selection.
- `artifact-manifest.json`, assigning the candidate version and file fingerprints.

The report records `promoted: false`. Export does not copy weights into
`severity_model`, change environment variables, rebuild containers, or publish
a model. Training failure can leave an incomplete export directory after I/O
starts; require a valid final artifact manifest before using it. Packages and
parents must stay immutable while the process reads them. Hashes verify contents
relative to local manifests, not trusted authorship or approved model quality.

The selected checkpoint is an inference export, not a resumable optimizer
checkpoint. Training cannot be resumed from an optimizer/scheduler state saved
by this command. The provenance is useful local experiment metadata, not a
complete environment lock or centralized experiment/model registry.

## Observed first learning run

The five-epoch CPU run exported `demo-severity-trained-v1`. Parent validation
accuracy/macro F1 were 0.2857/0.1481. All five trained epochs scored validation
accuracy/macro F1 of 0.4286/0.2000, so epoch 1 was selected. Training loss fell
from 1.0991 to 1.0759, but checkpoint selection followed validation macro F1.

The parent predicted Mild for all seven validation cases; the selected
candidate predicted Moderate for all seven. The modest score increase reflects
this class collapse on a tiny toy split, not useful severity discrimination.
No test score or clinical improvement is claimed. The active NER/severity
weight files were confirmed unchanged.

`tests/test_candidate_training.py` exercises package corruption/settings/output
guards and real repeatable training. It asserts no test/fixture text is scored,
checks that weights changed only in the candidate, verifies hashes/provenance,
and reloads the exported checkpoint through shared serving inference to compare
its validation metrics with the selected epoch.

Reference: [Transformers local model loading and saving](https://huggingface.co/docs/transformers/main_classes/model).
