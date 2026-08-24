from pathlib import Path

from PIL import Image

from src.data.prepare_products import build_products
from src.data.render_page import VIOLATIONS, render_one
from src.models.schema import AuditPrediction


def _config(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "source_mode": "synthetic_demo",
        "num_products": 2,
        "page_width": 960,
        "page_height": 720,
        "paths": {
            "products": (tmp_path / "products.jsonl").as_posix(),
            "raw_images": (tmp_path / "raw").as_posix(),
            "generated": (tmp_path / "generated").as_posix(),
            "manifests": (tmp_path / "manifests").as_posix(),
        },
    }


def test_all_v1_violations_render_and_validate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first, donor = build_products(config)
    for index, violation in enumerate(VIOLATIONS):
        row = render_one(first, donor, violation, "train", f"sample_{index}", index % 4, config)
        assert Path(row["image"]).exists()
        with Image.open(row["image"]) as image:
            assert image.size == (960, 720)
        AuditPrediction.model_validate(
            {key: row[key] for key in ("schema_version", "decision", "violation_type", "field", "listed_value", "observed_value", "evidence")}
        )
        if violation == "PASS":
            assert row["decision"] == "pass" and not row["evidence"]
        else:
            assert row["transform"] and row["evidence"]


def test_attribute_conflict_changes_only_recorded_field(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first, donor = build_products(config)
    row = render_one(first, donor, "ATTRIBUTE_CONFLICT", "train", "attribute", 0, config)
    assert set(row["changed_fields"]) == {"model"}
    assert row["listed_value"] != row["observed_value"]

