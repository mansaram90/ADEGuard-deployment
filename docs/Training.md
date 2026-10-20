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
That training command did not score the test split or establish clinical
improvement. The active NER/severity weight files were confirmed unchanged.

`tests/test_candidate_training.py` exercises package corruption/settings/output
guards and real repeatable training. It asserts no test/fixture text is scored,
checks that weights changed only in the candidate, verifies hashes/provenance,
and reloads the exported checkpoint through shared serving inference to compare
its validation metrics with the selected epoch.

Reference: [Transformers local model loading and saving](https://huggingface.co/docs/transformers/main_classes/model).

## Compare the frozen candidate with its parent

After selection, assess the fixed candidate and its recorded parent on the
same reserved six-case test split:

```powershell
python -m scripts.compare_severity_models --dataset outputs/datasets/severity/20261020T094655271040Z --candidate outputs/candidates/20261020T100542022874Z/model
```

`--parent` defaults to the configured severity model; if that changes later,
supply the original parent path explicitly. Optional `--evaluation` chooses
the protected synthetic fixture used during data preparation; it is verified,
not scored here. `--output` chooses a new report path. Defaults write a new
timestamped report under ignored `outputs/evaluation/`. Existing report files
and paths inside dataset/model packages are refused, including active model
directories. Inputs must remain immutable during assessment.

The command verifies the saved package and both model artifact manifests. The
candidate's hashed training report must identify the supplied parent/version,
the exact dataset manifest/source/split file hashes and counts, reserved test/
fixture flags, and the expected validation-only selection rule. Its best epoch
and selected metrics must agree with the recorded epoch history. These are
consistency checks on local recorded lineage, not proof of trusted provenance
or an audit of arbitrary external training behavior.

Both models then load locally on CPU, switch to evaluation mode, and predict
the identical ordered test examples through the shared serving method. The
trainer is never called. No epoch is selected again and no model/report files
inside either artifact are written. Thread settings are restored afterward.

The separate JSON report records model identities, selected epoch/training
report hash, dataset/test fingerprints, runtime and implementation hashes,
accuracy, macro F1, all per-class precision/recall/F1/support and confusion
matrices. Case IDs, expected/predicted labels and confidence show mistakes
without copying input text. Candidate-minus-parent deltas and paired outcome
IDs identify gains and regressions on the same examples.

### Observed first reserved-test assessment

| Metric | Parent | Fixed candidate |
|---|---:|---:|
| Correct cases | 2/6 | 3/6 |
| Accuracy | 0.3333 | 0.5000 |
| Macro F1 | 0.1667 | 0.2222 |
| Severe recall | 0.0000 | 0.0000 |
| Moderate recall | 0.0000 | 1.0000 |
| Mild recall | 1.0000 | 0.0000 |

The parent predicted Mild for every test case; the candidate predicted Moderate
for every case. The candidate gained three Moderate cases but lost two Mild
cases, and both missed the one Severe case. No case was correct for both.
These paired outcomes explain the aggregate score increase: one extra correct
answer, not evidence of reliable class discrimination. Tiny synthetic sample
counts do not establish statistical reliability or clinical performance.

The candidate remains epoch 1, and the original training report is unchanged.
Its `test_split_scored: false` describes the training/selection stage. The new
comparison report records the later test assessment separately. It applies no
quality gate, changes no serving selection, and records no promotion. A human
promotion decision is a separate step; a higher macro F1 alone does not make
this one-class candidate qualified for real use.

The test split is now observed. Keep the checkpoint choice fixed; using these
results to choose another epoch or tune settings would turn the test into a
selection set. Repeating the same frozen assessment for technical verification
does not add independent evidence. Future genuine selection requires suitable
validation data and a separate untouched benchmark.

`tests/test_model_comparison.py` rejects incorrect parent/data/selection lineage,
checks report/output guards, verifies only the same test texts are scored by
both models, checks case/paired outcomes, and compares all input file hashes
before/after the real assessment. Generated reports remain local.

## Record a HOLD decision without changing serving state

The review outcome for this synthetic candidate is **hold candidate; retain the
existing parent demo selection**. The candidate predicts only Moderate, misses
the Severe case and loses Mild recall, despite its higher aggregate score.
Six synthetic examples do not qualify either artifact for real-world use.
Retention preserves an operational learning baseline; it is not approval of
the parent's predictive quality.

Record the reasons and evidence rather than treating the metric increase as
an automatic deployment instruction:

```powershell
python -m scripts.record_model_decision --comparison outputs/evaluation/comparison-20261020T103147533338Z.json --dataset outputs/datasets/severity/20261020T094655271040Z --candidate outputs/candidates/20261020T100542022874Z/model --reason "One-class predictions, a missed Severe case, and loss of Mild recall on six synthetic cases."
```

Repeat `--reason` to record multiple explicit rationales. Use `--parent` if the
configured model no longer matches the candidate's recorded parent. Optional
`--evaluation` supplies the protected fixture for package verification and
`--output` selects a new record path. Existing reports or paths inside data/
model packages are refused. Defaults create a new timestamped JSON file under
ignored `outputs/decisions/`; there is no promote/activate operation.

The recorder verifies the prepared package, both artifact manifests and
training lineage, then binds the comparison to the same test IDs/labels,
dataset/test fingerprints, model identities, selected epoch and training-report
hash. It checks correctness flags and finite confidence, recalculates metrics
from the recorded predictions, and validates class/mistake counts. A mismatch
fails rather than recording a misleading score summary. Reasons must be
nonempty and distinct. The comparison SHA-256 covers the exact bytes parsed.

The generated decision contains the HOLD outcome, rationale, report fingerprint,
retained-parent/held-candidate identities, dataset/selection references and
verified metric summaries. It explicitly identifies an
`agent_assisted_manual_learning_review`, with clinical/production approval,
quality-gate application, inference, serving change and promotion all false.
The actual approved review used three reasons: one-class Moderate predictions;
zero Severe recall plus loss of Mild recall; and insufficient evidence from
six synthetic cases.

This is a recorded review judgment, not a predeclared automated quality policy
or a medical/production sign-off. The command checks consistency of local
recorded evidence; it does not rerun inference, authenticate prediction history,
prove reviewer identity, query the running API, or establish trusted authorship
of an unsigned comparison/manifest. It copies no weights and changes no API
configuration, environment selection or containers. Keep the original comparison,
artifacts and dataset available when reviewing the fingerprinted record.

`tests/test_model_decision.py` covers corrupted evidence, explicit reasons,
output protection and a successful fresh-process CLI run that loads neither
torch nor transformers and preserves all supplied input hashes. Existing
pytest CI discovers these tests without workflow changes. Actual records and
local `steps.md` remain outside the commit.

## Rehearse activation and rollback in an isolated preview

After recording HOLD, a separate operational exercise can demonstrate loading
and restoring exact packages. This is permission to test the synthetic candidate
in a disposable local preview, not permission to promote it. The quality HOLD
remains in force. No further fitting, checkpoint selection or benchmark scoring
is performed.

`compose.model-preview.yaml` runs only an API on `127.0.0.1:18000`, using a
digest-pinned published API image. A read-only bind mount selects the severity
package at `/models/severity`; `ADEGUARD_SEVERITY_MODEL_PATH` tells the existing
startup loader to use it. NER remains inside the image. `create_host_path: false`
rejects missing host directories. Explicit `pull_policy: never` and `--no-build`
keep the image fixed during the drill; pull that exact image first if absent.
The separate project name avoids recreating the normal API/frontend stack.
Run this file alone, without combining it with development/release Compose.

From the repository root, create two local selections (use the actual immutable
candidate path from your own training run if it differs):

```powershell
New-Item -ItemType Directory -Force outputs/model-preview | Out-Null
$previewDigest = '39552a3ef0fa6abaf7b3cb4f65101fa6f5d9e9ac475b446848f2157aac7ac134'
$previewParent = (Resolve-Path severity_model).Path.Replace('\', '/')
$previewCandidate = (Resolve-Path outputs/candidates/20261020T100542022874Z/model).Path.Replace('\', '/')
@("ADEGUARD_PREVIEW_API_DIGEST=$previewDigest", "ADEGUARD_PREVIEW_MODEL_DIR=$previewParent") | Set-Content outputs/model-preview/parent.env -Encoding ascii
@("ADEGUARD_PREVIEW_API_DIGEST=$previewDigest", "ADEGUARD_PREVIEW_MODEL_DIR=$previewCandidate") | Set-Content outputs/model-preview/candidate.env -Encoding ascii
docker compose --env-file outputs/model-preview/parent.env -f compose.model-preview.yaml config --quiet
# Download the fixed image if it is not already cached:
docker compose --env-file outputs/model-preview/parent.env -f compose.model-preview.yaml pull
```

The recorded digest is the previously tested API image published for commit
`34e157130d4ba851f257019e859ac8de2ad780f0`; it is not a claim about the latest
image. This drill changes model selection independently of the serving code.
Keep packages immutable throughout. The example setup overwrites the two local
preview env selections, so inspect existing selections before recreating them.

Run each verification only after its preceding Compose command succeeds:

```powershell
# 1. Establish the parent baseline.
docker compose --env-file outputs/model-preview/parent.env -f compose.model-preview.yaml up -d --no-build --pull never --wait --wait-timeout 240
python -m scripts.check_model_preview --model severity_model --stage parent

# 2. Recreate only the preview API with the candidate mount.
docker compose --env-file outputs/model-preview/candidate.env -f compose.model-preview.yaml up -d --no-build --pull never --wait --wait-timeout 240
python -m scripts.check_model_preview --model outputs/candidates/20261020T100542022874Z/model --stage candidate

# 3. Restore the parent and verify its identity again.
docker compose --env-file outputs/model-preview/parent.env -f compose.model-preview.yaml up -d --no-build --pull never --wait --wait-timeout 240
python -m scripts.check_model_preview --model severity_model --stage rollback

# 4. Remove only the disposable preview project.
docker compose --env-file outputs/model-preview/parent.env -f compose.model-preview.yaml down --remove-orphans
```

If candidate startup or verification fails, stop the sequence and run the
parent restoration plus rollback check before cleanup. Use the same explicit
Compose file on all commands; do not run `down` against the normal stack.
For diagnosis, run `docker compose --env-file outputs/model-preview/parent.env
-f compose.model-preview.yaml logs --tail 80 api` (on one line).

`scripts/check_model_preview.py` uses only standard-library HTTP plus existing
artifact verification and smoke-response validation. It verifies local package
integrity, `/ready`, exact severity and NER version/type/manifest identities,
the preview severity mount path, and one `/predict` response's structure and
probabilities. It rechecks local identities afterward and saves a new timestamped
JSON observation under ignored `outputs/model-preview/`. It does not change
containers, choose an image, perform promotion, or infer approval from a stage
name. The `--stage` label describes the operator's step; individual reports must
be reviewed together to establish the parent/candidate/parent sequence. Errors
return a nonzero status without writing a success report.

The same fixed synthetic request exercises serving, not model quality. Readiness
alone cannot establish which model loaded. A successful rollback establishes a
local recovery procedure, not zero-downtime deployment or a measured recovery
SLA: each switch recreates one API container and temporarily interrupts preview
requests. The normal frontend still targets port 8000; this API-only exercise
does not test browser integration. Unsigned manifest identity proves package
consistency, not publisher authenticity. The HTTP checker does not attest the
container image digest; Compose's pinned image selects the executable, while
the checker verifies model identity reported by that executable.

`tests/test_model_preview.py` checks readiness/identity/mount/prediction failures,
package mutation, failed-CLI output behavior and separate report creation.
The normal backend CI discovers these unit tests; the live Docker drill is
manual and uses local generated packages, so no CI workflow changes are needed.
