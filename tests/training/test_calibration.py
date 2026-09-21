import pytest
import torch

from tensorcode.training.calibration import TemperatureCalibration, evaluate_calibration, fit_threshold


def test_temperature_fit_improves_nll_preserves_predictions_and_model():
    model = torch.nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.eye(2) * 12)
    logits = model(torch.eye(2).repeat(2, 1).double())
    labels = torch.tensor([0, 1, 1, 0])
    calibration = TemperatureCalibration()
    before = model.weight.detach().clone()
    assert not calibration.calibrated.item()
    report = calibration.fit(logits, labels)
    assert report['after']['nll'] < report['before']['nll']
    assert torch.equal(calibration(logits).argmax(-1), logits.argmax(-1))
    assert torch.equal(before, model.weight)
    assert model.weight.grad is None
    assert calibration.sample_count.item() == 4
    assert calibration.calibrated.item()
    restored = TemperatureCalibration(**calibration.configuration())
    restored.load_state_dict(calibration.state_dict())
    assert torch.equal(restored(logits), calibration(logits))
    assert restored.calibrated.item()
    repeated = TemperatureCalibration()
    repeated.fit(logits, labels)
    assert torch.equal(repeated.temperature, calibration.temperature)


@pytest.mark.parametrize('logits,labels', [
    (torch.tensor([[float('nan'), 0.]]), torch.tensor([0])),
    (torch.ones(0, 2), torch.tensor([], dtype=torch.long)),
    (torch.ones(2, 2), torch.tensor([0])),
    (torch.ones(2, 2), torch.tensor([0., 1.])),
    (torch.ones(2, 2), torch.tensor([0, 2])),
])
def test_invalid_fit_does_not_mutate(logits, labels):
    calibration = TemperatureCalibration()
    original = {k: v.clone() for k, v in calibration.state_dict().items()}
    with pytest.raises((ValueError, TypeError)):
        calibration.fit(logits, labels)
    assert all(torch.equal(v, original[k]) for k, v in calibration.state_dict().items())


def test_metrics_and_bin_validation():
    result = evaluate_calibration(torch.zeros(2, 2), torch.tensor([0, 1]), n_bins=2)
    assert result['nll'] == pytest.approx(0.69314718)
    assert result['brier'] == pytest.approx(0.5)
    assert result['accuracy'] == pytest.approx(0.5)
    assert result['ece'] == pytest.approx(0.)
    with pytest.raises(ValueError):
        evaluate_calibration(torch.zeros(2, 2), torch.tensor([0, 1]), n_bins=0)


def test_threshold_selects_largest_empirical_coverage_without_splitting_ties():
    report = fit_threshold(torch.tensor([.9, .8, .8, .6]), torch.tensor([True, True, False, False]), max_error=0.)
    assert report['accepted_count'] == 1
    assert report['coverage'] == .25
    assert report['error'] == 0.
    assert report['sample_count'] == 4
    empty = fit_threshold(torch.tensor([1.]), torch.tensor([False]), max_error=0.)
    assert empty['threshold'] is None
    assert empty['accepted_count'] == 0
    assert empty['error'] is None
    with pytest.raises(ValueError):
        fit_threshold(torch.tensor([1.1]), torch.tensor([True]), max_error=0.)
