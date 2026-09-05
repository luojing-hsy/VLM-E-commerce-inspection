import json
from pathlib import Path

import pytest

from src.common import write_jsonl
from src.data.export_verl_sft import build_rows
from src.models.audit_protocol import (
    OUTPUT_KEYS,
    product_prompt,
    prompt_with_image_token,
    structured_prompt,
    validate_prediction_dict,
)
from src.rewards.parser import tolerant_parse


def _pass_target() -> dict:
    return {
        "decision": "pass",
        "violation_type": "pass",
        "issue_subtype": None,
        "evidence": None,
    }


def test_protocol_is_strict_four_field_json() -> None:
    assert tuple(validate_prediction_dict(_pass_target())) == OUTPUT_KEYS
    with pytest.raises(ValueError, match="exactly"):
        validate_prediction_dict({**_pass_target(), "sample_id": "secret"})
    assert not tolerant_parse(f"```json\n{json.dumps(_pass_target())}\n```").protocol_valid


def test_sft_prompt_places_images_before_text_fields() -> None:
    prompt = product_prompt("Sample title", "sample_category", None, "fabric", image_placeholders=True)
    assert prompt.index("main: <image>") < prompt.index("title:")
    assert (
        prompt.index("main: <image>")
        < prompt.index("detail:1: <image>")
        < prompt.index("detail:2: <image>")
    )


def test_structured_prompt_role_tags_each_image_in_sft_order(tmp_path: Path) -> None:
    images = [tmp_path / f"image_{index}.png" for index in range(3)]
    text = product_prompt("Sample title", "sample_category", None, "fabric", image_placeholders=False)

    content = structured_prompt([image.as_posix() for image in images], text)[0]["content"]

    assert [part["type"] for part in content] == ["text", "image", "text", "image", "text", "image", "text"]
    assert [content[index]["text"] for index in (0, 2, 4)] == [
        "main: ",
        "\ndetail:1: ",
        "\ndetail:2: ",
    ]
    assert content[-1]["text"] == f"\n{text}"
    assert [content[index]["image"] for index in (1, 3, 5)] == [str(image) for image in images]


def test_sft_parquet_rows_expose_four_fields_and_three_images(tmp_path: Path) -> None:
    images = [tmp_path / f"image_{index}.png" for index in range(3)]
    for image in images:
        image.write_bytes(b"image")
    source = tmp_path / "sft_train.jsonl"
    prompt = product_prompt("Sample title", "sample_category", None, "fabric", image_placeholders=True)
    write_jsonl(
        source,
        [
            {
                "sample_id": "do-not-expose-this-id",
                "dataset_stage": "sft",
                "split": "train",
                "images": [image.as_posix() for image in images],
                "lineage": {"transform": "do-not-expose-this-transform"},
                "conversations": [
                    {"from": "human", "value": prompt},
                    {
                        "from": "gpt",
                        "value": json.dumps(_pass_target(), ensure_ascii=False, separators=(",", ":")),
                    },
                ],
            }
        ],
    )
    rows = build_rows(source)
    assert set(rows[0]) == {"messages", "images"}
    assert rows[0]["images"] == [image.as_posix() for image in images]
    assert rows[0]["messages"][0]["content"].count("<image>") == 3
    visible = json.dumps(rows[0]["messages"], ensure_ascii=False)
    assert "do-not-expose-this-id" not in visible
    assert "do-not-expose-this-transform" not in visible

