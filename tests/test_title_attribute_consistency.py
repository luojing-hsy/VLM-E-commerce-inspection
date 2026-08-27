import pytest

from scripts.build_synthesis_datasets import (
    TitleMutationError,
    adjust_quota_for_title_eligibility,
    mutate_title_attribute,
    replace_text,
)


def test_replace_text_replaces_all_occurrences() -> None:
    assert replace_text("Tan cover, tan trim", "Tan", "Dark Blue") == "Dark Blue cover, Dark Blue trim"


def test_mutate_title_replaces_hyphenated_and_slash_mentions() -> None:
    title = "Stainless-Steel case in Black/Grey"
    mutated, audit = mutate_title_attribute(
        title,
        "material",
        "Stainless Steel",
        "Glass",
    )
    assert mutated == "Glass case in Black/Grey"
    assert audit["source_mentions"] == 1
    assert audit["replaced_mentions"] == 1


def test_mutate_title_accepts_ampersand_and_word_separator_variants() -> None:
    mutated, audit = mutate_title_attribute(
        "Black and Gray organizer",
        "color",
        "Black & Gray",
        "Blue/White",
    )
    assert mutated == "Blue/White organizer"
    assert audit["source_mentions"] == 1


def test_mutate_title_rejects_unordered_multitoken_mentions() -> None:
    with pytest.raises(TitleMutationError, match="ambiguous"):
        mutate_title_attribute(
            "Nude Beige [W3] powder",
            "color",
            "W3 Nude Beige",
            "Black",
        )


def test_mutate_title_records_when_target_is_not_in_title() -> None:
    title = "Plain product name"
    mutated, audit = mutate_title_attribute(title, "color", "Tan", "Blue")
    assert mutated == title
    assert audit["policy"] == "no_mention"
    assert audit["source_mentions"] == 0


def test_quota_shortfall_moves_to_pass() -> None:
    rows = [{"product_id": "a"}, {"product_id": "b"}]
    raw_by_id = {
        "a": {"title": "Plain cover", "category": "cover", "color": "Tan", "material": None},
        "b": {"title": "Nude Beige [W3] powder", "category": "powder", "color": "W3 Nude Beige", "material": None},
    }
    quota = {
        "pass": 0,
        "duplicate_detail_image": 1,
        "image_quality": 1,
        "category_mismatch": 1,
        "wrong_image": 0,
        "color_mismatch": 2,
        "title_mismatch": 1,
        "material_mismatch": 0,
    }
    effective, adjustments = adjust_quota_for_title_eligibility(rows, raw_by_id, quota)
    assert effective["color_mismatch"] == 1
    assert effective["pass"] == 1
    assert adjustments["color_mismatch"] == {
        "requested": 2,
        "eligible_capacity": 1,
        "moved_to_pass": 1,
    }

