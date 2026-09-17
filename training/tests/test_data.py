import numpy as np

from training.data import stratified_indices


def test_stratified_indices_are_deterministic_disjoint_and_balanced() -> None:
    targets = np.repeat(np.arange(10), 20)

    train_a, validation_a = stratified_indices(targets, validation_fraction=0.2, seed=42)
    train_b, validation_b = stratified_indices(targets, validation_fraction=0.2, seed=42)

    assert train_a == train_b
    assert validation_a == validation_b
    assert set(train_a).isdisjoint(validation_a)
    assert len(train_a) == 160
    assert len(validation_a) == 40
    assert np.bincount(targets[validation_a], minlength=10).tolist() == [4] * 10

