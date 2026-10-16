import pytest

from src.adeguard_inference import InputValidationError, NERModel, SeverityModel


@pytest.fixture(scope="module", params=[NERModel, SeverityModel])
def inference_runner(request):
    model = request.param.load()
    predict = model.predict_one if isinstance(model, NERModel) else model.predict
    return model, predict


def test_model_accepts_exact_token_capacity(inference_runner):
    model, predict = inference_runner
    # "patient" is one token in the demo vocabulary; BERT adds CLS and SEP.
    limit = model.model.config.max_position_embeddings
    text = " ".join(["patient"] * (limit - 2))
    assert len(model.tokenizer(text)["input_ids"]) == limit
    assert predict(text) is not None


def test_model_rejects_one_token_over_capacity_before_forward(inference_runner, monkeypatch):
    model, predict = inference_runner
    text = " ".join(["patient"] * (model.model.config.max_position_embeddings - 1))

    def unexpected_forward(*args, **kwargs):
        pytest.fail("Oversized input reached the model forward pass")

    monkeypatch.setattr(model.model, "forward", unexpected_forward)
    with pytest.raises(InputValidationError, match="including special tokens"):
        predict(text)


def test_model_rejects_whitespace(inference_runner):
    _, predict = inference_runner
    with pytest.raises(InputValidationError, match="non-whitespace"):
        predict(" \n\t ")


def test_severity_model_returns_three_probabilities():
    model = SeverityModel.load()

    prediction = model.predict("Patient reported fever after Pfizer booster.")

    assert prediction.label in {"Severe", "Moderate", "Mild"}
    assert len(prediction.scores) == 3
    assert set(prediction.probabilities) == {"Severe", "Moderate", "Mild"}
    assert 0.0 <= prediction.confidence <= 1.0


def test_ner_model_returns_expected_response_shape():
    model = NERModel.load()

    prediction = model.predict_one("Patient reported fever after Pfizer booster.")

    assert isinstance(prediction.ade, list)
    assert isinstance(prediction.drug, list)
    assert isinstance(prediction.highlights, list)
    assert all(len(highlight) == 3 for highlight in prediction.highlights)
