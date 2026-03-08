#!/usr/bin/env python3
"""Problem 4.1: Bayesian Sensor Reasoning calculator.

This script computes all requested posterior probabilities for:
1) No creaking observed (single-square assumption)
2) Creaking observed (single-square assumption)
3) Sensor quality comparison with a second sensor
4) Multi-adjacent-square case for P(D1=1 | C=0)
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse


@dataclass(frozen=True)
class SensorModel:
    true_positive: float
    false_positive: float

    @property
    def false_negative(self) -> float:
        return 1.0 - self.true_positive

    @property
    def true_negative(self) -> float:
        return 1.0 - self.false_positive


def posterior_single_square(prior_damage: float, sensor: SensorModel, observed_creak: bool) -> dict[str, float]:
    """Compute Bayes posterior P(D=1 | C=c) for one adjacent square."""
    if observed_creak:
        likelihood_if_damage = sensor.true_positive
        likelihood_if_no_damage = sensor.false_positive
    else:
        likelihood_if_damage = sensor.false_negative
        likelihood_if_no_damage = sensor.true_negative

    evidence = (
        likelihood_if_damage * prior_damage
        + likelihood_if_no_damage * (1.0 - prior_damage)
    )
    posterior = (likelihood_if_damage * prior_damage) / evidence

    return {
        "prior": prior_damage,
        "likelihood_if_damage": likelihood_if_damage,
        "likelihood_if_no_damage": likelihood_if_no_damage,
        "evidence": evidence,
        "posterior": posterior,
    }


def posterior_multi_adjacent_d1_given_no_creak(prior_damage: float, sensor: SensorModel) -> dict[str, float]:
    """Compute P(D1=1 | C=0) when C reflects at least one adjacent damage (D1 or D2)."""
    p_d1 = prior_damage
    p_d2 = prior_damage

    p_at_least_one_damaged = 1.0 - (1.0 - p_d1) * (1.0 - p_d2)

    p_c0_given_at_least_one = sensor.false_negative
    p_c0_given_none = sensor.true_negative

    p_c0 = (
        p_c0_given_at_least_one * p_at_least_one_damaged
        + p_c0_given_none * (1.0 - p_at_least_one_damaged)
    )

    p_c0_given_d1 = sensor.false_negative

    p_d2_given_not_d1 = prior_damage
    p_c0_given_not_d1 = (
        sensor.false_negative * p_d2_given_not_d1
        + sensor.true_negative * (1.0 - p_d2_given_not_d1)
    )

    p_c0_check = p_c0_given_d1 * p_d1 + p_c0_given_not_d1 * (1.0 - p_d1)

    posterior = (p_c0_given_d1 * p_d1) / p_c0

    return {
        "p_at_least_one_damaged": p_at_least_one_damaged,
        "p_c0_given_at_least_one": p_c0_given_at_least_one,
        "p_c0_given_none": p_c0_given_none,
        "evidence_p_c0": p_c0,
        "p_c0_given_d1": p_c0_given_d1,
        "p_c0_given_not_d1": p_c0_given_not_d1,
        "evidence_check_from_d1_partition": p_c0_check,
        "posterior_p_d1_given_c0": posterior,
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def print_single_square_result(title: str, result: dict[str, float], observed_creak: bool) -> None:
    event_text = "C=1 (creaking observed)" if observed_creak else "C=0 (no creaking)"
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Event: {event_text}")
    print(f"Prior P(D=1): {fmt(result['prior'])}")
    print(f"Likelihood P(C|D=1): {fmt(result['likelihood_if_damage'])}")
    print(f"Likelihood P(C|D=0): {fmt(result['likelihood_if_no_damage'])}")
    print(f"Evidence P(C): {fmt(result['evidence'])}")
    print(f"Posterior P(D=1|C): {fmt(result['posterior'])}")


def print_multi_square_result(result: dict[str, float]) -> None:
    title = "Part 4 - Multiple adjacent squares"
    print(f"\n{title}")
    print("-" * len(title))
    print("Goal: P(D1=1 | C=0)")
    print(f"P(at least one damaged): {fmt(result['p_at_least_one_damaged'])}")
    print(f"P(C=0 | at least one damaged): {fmt(result['p_c0_given_at_least_one'])}")
    print(f"P(C=0 | none damaged): {fmt(result['p_c0_given_none'])}")
    print(f"Evidence P(C=0): {fmt(result['evidence_p_c0'])}")
    print(f"P(C=0 | D1=1): {fmt(result['p_c0_given_d1'])}")
    print(f"P(C=0 | D1=0): {fmt(result['p_c0_given_not_d1'])}")
    print(f"Evidence cross-check: {fmt(result['evidence_check_from_d1_partition'])}")
    print(f"Posterior P(D1=1 | C=0): {fmt(result['posterior_p_d1_given_c0'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian sensor reasoning for Problem 4.1")
    parser.add_argument("--prior", type=float, default=0.15, help="Prior probability of damage")
    parser.add_argument("--tp", type=float, default=0.85, help="True-positive rate of base sensor")
    parser.add_argument("--fp", type=float, default=0.08, help="False-positive rate of base sensor")
    parser.add_argument("--tp2", type=float, default=0.95, help="True-positive rate of newer sensor")
    parser.add_argument("--fp2", type=float, default=0.02, help="False-positive rate of newer sensor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sensor_base = SensorModel(true_positive=args.tp, false_positive=args.fp)
    sensor_new = SensorModel(true_positive=args.tp2, false_positive=args.fp2)

    print("Problem 4.1 Bayesian Sensor Reasoning")
    print("=" * 40)

    part1 = posterior_single_square(args.prior, sensor_base, observed_creak=False)
    print_single_square_result("Part 1 - Base sensor, no creaking", part1, observed_creak=False)

    part2 = posterior_single_square(args.prior, sensor_base, observed_creak=True)
    print_single_square_result("Part 2 - Base sensor, creaking observed", part2, observed_creak=True)

    part3a = posterior_single_square(args.prior, sensor_new, observed_creak=False)
    print_single_square_result("Part 3A - New sensor, no creaking", part3a, observed_creak=False)

    part3b = posterior_single_square(args.prior, sensor_new, observed_creak=True)
    print_single_square_result("Part 3B - New sensor, creaking observed", part3b, observed_creak=True)

    part4 = posterior_multi_adjacent_d1_given_no_creak(args.prior, sensor_base)
    print_multi_square_result(part4)


if __name__ == "__main__":
    main()
