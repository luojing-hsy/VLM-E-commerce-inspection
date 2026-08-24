from collections import Counter
from pathlib import Path

from PIL import Image
import pytest

from src.data.prepare_abo import _phash64
from src.data.split_highres_dataset import Component, _phash64_with_compatibility, assign_components


def test_component_assignment_is_exact_stable_and_leakage_safe() -> None:
    components = [
        Component("paired-a", (0, 1), Counter({"a": 2})),
        Component("paired-b", (2, 3), Counter({"b": 2})),
        *[
            Component(f"single-{index}", (index,), Counter({"a" if index % 2 else "b": 1}))
            for index in range(4, 20)
        ],
    ]
    targets = {"left": 8, "middle": 5, "right": 7}
    strata = Counter({"a": 10, "b": 10})

    first = assign_components(components, strata, seed=42, target_sizes=targets)
    second = assign_components(components, strata, seed=42, target_sizes=targets)

    assert first == second
    counts = Counter()
    for component in components:
        counts[first[component.component_id]] += component.size
    assert counts == Counter(targets)
    assert first["paired-a"] in targets
    assert first["paired-b"] in targets


def test_phash_retries_only_for_truncated_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "truncated.jpg"
    Image.new("RGB", (256, 256), "navy").save(path, quality=95)
    payload = path.read_bytes()
    path.write_bytes(payload[:-32])

    with pytest.raises(OSError, match="truncated"):
        _phash64(path)

    phash, used_compatibility = _phash64_with_compatibility(path)
    assert isinstance(phash, int)
    assert used_compatibility is True
