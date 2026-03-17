import pytest

from src.bayesian_sensor_reasoning import (
    SensorModel,
    posterior_multi_adjacent_d1_given_no_creak,
    posterior_single_square,
)


def test_base_sensor_no_creaking_posterior() -> None:
    sensor = SensorModel(true_positive=0.85, false_positive=0.08)
    result = posterior_single_square(0.15, sensor, observed_creak=False)

    assert result["evidence"] == pytest.approx(0.8045, rel=1e-9, abs=1e-12)
    assert result["posterior"] == pytest.approx(0.027967681790, rel=1e-9, abs=1e-12)


def test_base_sensor_creaking_posterior() -> None:
    sensor = SensorModel(true_positive=0.85, false_positive=0.08)
    result = posterior_single_square(0.15, sensor, observed_creak=True)

    assert result["evidence"] == pytest.approx(0.1955, rel=1e-9, abs=1e-12)
    assert result["posterior"] == pytest.approx(0.652173913043, rel=1e-9, abs=1e-12)


def test_new_sensor_comparison_values() -> None:
    sensor_new = SensorModel(true_positive=0.95, false_positive=0.02)
    no_creak = posterior_single_square(0.15, sensor_new, observed_creak=False)
    creak = posterior_single_square(0.15, sensor_new, observed_creak=True)

    assert no_creak["posterior"] == pytest.approx(0.008922070196, rel=1e-9, abs=1e-12)
    assert creak["posterior"] == pytest.approx(0.893416927900, rel=1e-9, abs=1e-12)


def test_total_probability_consistency() -> None:
    sensor = SensorModel(true_positive=0.85, false_positive=0.08)
    no_creak = posterior_single_square(0.15, sensor, observed_creak=False)
    creak = posterior_single_square(0.15, sensor, observed_creak=True)

    reconstructed_prior = no_creak["posterior"] * no_creak["evidence"] + creak["posterior"] * creak["evidence"]
    assert reconstructed_prior == pytest.approx(0.15, rel=1e-9, abs=1e-12)


def test_multi_adjacent_square_posterior() -> None:
    sensor = SensorModel(true_positive=0.85, false_positive=0.08)
    result = posterior_multi_adjacent_d1_given_no_creak(0.15, sensor)

    assert result["p_at_least_one_damaged"] == pytest.approx(0.2775, rel=1e-9, abs=1e-12)
    assert result["evidence_p_c0"] == pytest.approx(0.706325, rel=1e-9, abs=1e-12)
    assert result["posterior_p_d1_given_c0"] == pytest.approx(0.031854572303, rel=1e-9, abs=1e-12)
