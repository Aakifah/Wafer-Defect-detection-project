"""Comprehensive tests for data deduplication, stratified splitting, and integrity."""

import numpy as np
import numpy.typing as npt
import pytest

from src.ETL_data.ingestion import (
    deduplicate,
    stratified_split,
    verify_no_cross_contamination,
)


def _make_sample_data(
    n_samples: int = 100,
    n_classes: int = 8,
    map_size: int = 52,
    seed: int = 42,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n_samples, map_size, map_size), dtype=np.int32)
    y = rng.integers(0, 2, size=(n_samples, n_classes), dtype=np.int32)
    return X, y


def _make_data_with_duplicates(
    n_unique: int = 80,
    n_dupes: int = 20,
    n_classes: int = 8,
    map_size: int = 52,
    seed: int = 42,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    rng = np.random.default_rng(seed)
    X_unique = rng.integers(0, 4, size=(n_unique, map_size, map_size), dtype=np.int32)
    y_unique = rng.integers(0, 2, size=(n_unique, n_classes), dtype=np.int32)

    dupe_indices = rng.choice(n_unique, size=n_dupes, replace=True)
    X_dupes = X_unique[dupe_indices].copy()
    y_dupes = y_unique[dupe_indices].copy()

    X = np.vstack([X_unique, X_dupes])
    y = np.vstack([y_unique, y_dupes])
    return X, y


class TestDeduplicate:
    def test_removes_exact_xy_duplicates(self) -> None:
        X, y = _make_data_with_duplicates(n_unique=80, n_dupes=20)
        X_dedup, y_dedup, removed = deduplicate(X, y)

        assert len(X_dedup) == 80
        assert len(y_dedup) == 80
        assert removed == 20

    def test_no_duplicates_remain_after_dedup(self) -> None:
        X, y = _make_data_with_duplicates()
        X_dedup, y_dedup, _ = deduplicate(X, y)

        X_flat = X_dedup.reshape(len(X_dedup), -1)
        combined = np.hstack([X_flat, y_dedup])
        unique_count = len(np.unique(combined, axis=0))
        assert unique_count == len(X_dedup)

    def test_preserves_data_integrity(self) -> None:
        X, y = _make_data_with_duplicates()
        X_dedup, y_dedup, _ = deduplicate(X, y)

        assert X_dedup.dtype == X.dtype
        assert y_dedup.dtype == y.dtype
        assert X_dedup.shape[1:] == X.shape[1:]
        assert y_dedup.shape[1] == y.shape[1]
        assert X_dedup.min() >= 0 and X_dedup.max() <= 3
        assert np.all(np.isin(y_dedup, [0, 1]))

    def test_noop_when_no_duplicates(self) -> None:
        X, y = _make_sample_data(n_samples=50)
        X_dedup, y_dedup, removed = deduplicate(X, y)

        assert removed == 0
        assert len(X_dedup) == len(X)
        np.testing.assert_array_equal(X_dedup, X)
        np.testing.assert_array_equal(y_dedup, y)

    def test_single_sample(self) -> None:
        X = np.zeros((1, 52, 52), dtype=np.int32)
        y = np.array([[0, 1, 0, 0, 0, 0, 1, 0]], dtype=np.int32)
        X_dedup, y_dedup, removed = deduplicate(X, y)

        assert len(X_dedup) == 1
        assert removed == 0

    def test_all_identical_samples(self) -> None:
        X = np.zeros((100, 52, 52), dtype=np.int32)
        y = np.ones((100, 8), dtype=np.int32)
        X_dedup, y_dedup, removed = deduplicate(X, y)

        assert len(X_dedup) == 1
        assert removed == 99


class TestStratifiedSplit:
    def test_split_ratios_approximate(self) -> None:
        X, y = _make_sample_data(n_samples=1000)
        splits = stratified_split(X, y)

        total = len(X)
        n_train = len(splits["train"][0])
        n_val = len(splits["val"][0])
        n_test = len(splits["test"][0])

        assert n_train + n_val + n_test == total
        assert abs(n_train / total - 0.70) < 0.03
        assert abs(n_val / total - 0.15) < 0.03
        assert abs(n_test / total - 0.15) < 0.03

    def test_no_cross_contamination(self) -> None:
        X, y = _make_sample_data(n_samples=500)
        splits = stratified_split(X, y)

        X_train_flat = splits["train"][0].reshape(len(splits["train"][0]), -1)
        X_val_flat = splits["val"][0].reshape(len(splits["val"][0]), -1)
        X_test_flat = splits["test"][0].reshape(len(splits["test"][0]), -1)

        train_set = set(map(tuple, X_train_flat))
        val_set = set(map(tuple, X_val_flat))
        test_set = set(map(tuple, X_test_flat))

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_label_distribution_preserved(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.integers(0, 4, size=(1000, 52, 52), dtype=np.int32)
        y = np.zeros((1000, 8), dtype=np.int32)
        for c in range(8):
            n_pos = int(1000 * (0.2 + c * 0.05))
            y[:n_pos, c] = 1

        splits = stratified_split(X, y)

        overall_rates = y.mean(axis=0)
        for name, (_, y_split) in splits.items():
            split_rates = y_split.mean(axis=0)
            np.testing.assert_allclose(
                split_rates, overall_rates, atol=0.08,
                err_msg=f"Label distribution drifted in {name} split",
            )

    def test_deterministic_with_same_seed(self) -> None:
        X, y = _make_sample_data(n_samples=200)
        splits1 = stratified_split(X, y, random_state=42)
        splits2 = stratified_split(X, y, random_state=42)

        for key in splits1:
            np.testing.assert_array_equal(splits1[key][0], splits2[key][0])
            np.testing.assert_array_equal(splits1[key][1], splits2[key][1])

    def test_data_integrity_preserved(self) -> None:
        X, y = _make_sample_data(n_samples=300)
        splits = stratified_split(X, y)

        for name, (X_s, y_s) in splits.items():
            assert X_s.dtype == X.dtype
            assert y_s.dtype == y.dtype
            assert X_s.shape[1:] == X.shape[1:]
            assert y_s.shape[1] == y.shape[1]
            assert X_s.min() >= 0 and X_s.max() <= 3
            assert np.all(np.isin(y_s, [0, 1]))

    def test_all_splits_non_empty(self) -> None:
        X, y = _make_sample_data(n_samples=100)
        splits = stratified_split(X, y)

        for name, (X_s, y_s) in splits.items():
            assert len(X_s) > 0, f"{name} split is empty"
            assert len(y_s) > 0, f"{name} split is empty"

    def test_small_dataset(self) -> None:
        X, y = _make_sample_data(n_samples=20)
        splits = stratified_split(X, y)

        total = sum(len(s[0]) for s in splits.values())
        assert total == len(X)

    def test_multi_class_imbalance(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.integers(0, 4, size=(500, 52, 52), dtype=np.int32)
        y = np.zeros((500, 8), dtype=np.int32)
        y[0:5, 5] = 1
        y[5:15, 7] = 1
        y[15:, 0] = 1

        splits = stratified_split(X, y)

        for name, (_, y_s) in splits.items():
            assert y_s.shape[1] == 8
            assert len(y_s) > 0


class TestVerifyNoCrossContamination:
    def test_passes_with_clean_splits(self) -> None:
        rng = np.random.default_rng(42)
        X_train = rng.integers(0, 4, size=(100, 52, 52), dtype=np.int32)
        X_val = rng.integers(100, 104, size=(50, 52, 52), dtype=np.int32)
        X_test = rng.integers(200, 204, size=(50, 52, 52), dtype=np.int32)
        y_train = np.zeros((100, 8), dtype=np.int32)
        y_val = np.zeros((50, 8), dtype=np.int32)
        y_test = np.zeros((50, 8), dtype=np.int32)

        verify_no_cross_contamination(
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        )

    def test_detects_train_val_overlap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        exit_calls: list[int] = []
        monkeypatch.setattr("sys.exit", lambda code: exit_calls.append(code))

        X_train = np.zeros((10, 52, 52), dtype=np.int32)
        X_val = np.zeros((10, 52, 52), dtype=np.int32)
        X_test = np.ones((10, 52, 52), dtype=np.int32)
        y_train = np.zeros((10, 8), dtype=np.int32)
        y_val = np.zeros((10, 8), dtype=np.int32)
        y_test = np.ones((10, 8), dtype=np.int32)

        verify_no_cross_contamination(
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        )
        assert exit_calls == [1]

    def test_detects_train_test_overlap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        exit_calls: list[int] = []
        monkeypatch.setattr("sys.exit", lambda code: exit_calls.append(code))

        X_train = np.zeros((10, 52, 52), dtype=np.int32)
        X_val = np.ones((10, 52, 52), dtype=np.int32)
        X_test = np.zeros((10, 52, 52), dtype=np.int32)
        y_train = np.zeros((10, 8), dtype=np.int32)
        y_val = np.ones((10, 8), dtype=np.int32)
        y_test = np.zeros((10, 8), dtype=np.int32)

        verify_no_cross_contamination(
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        )
        assert exit_calls == [1]


class TestEndToEndPipeline:
    def test_dedup_then_split_no_contamination(self) -> None:
        X, y = _make_data_with_duplicates(n_unique=200, n_dupes=50)
        X_dedup, y_dedup, _ = deduplicate(X, y)
        splits = stratified_split(X_dedup, y_dedup)

        verify_no_cross_contamination(
            splits["train"], splits["val"], splits["test"]
        )

    def test_dedup_then_split_preserves_total(self) -> None:
        X, y = _make_data_with_duplicates(n_unique=200, n_dupes=50)
        X_dedup, y_dedup, removed = deduplicate(X, y)
        splits = stratified_split(X_dedup, y_dedup)

        total_in_splits = sum(len(s[0]) for s in splits.values())
        assert total_in_splits == len(X_dedup)
        assert total_in_splits == len(X) - removed

    def test_all_samples_accounted_for(self) -> None:
        X, y = _make_sample_data(n_samples=500)
        X_dedup, y_dedup, _ = deduplicate(X, y)
        splits = stratified_split(X_dedup, y_dedup)

        all_X = np.vstack([s[0] for s in splits.values()])
        all_y = np.vstack([s[1] for s in splits.values()])

        assert len(all_X) == len(X_dedup)
        assert len(all_y) == len(y_dedup)
