from src.adeguard_inference import NERModel, SeverityModel


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
