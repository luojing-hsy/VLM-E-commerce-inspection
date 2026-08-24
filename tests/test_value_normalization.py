from src.rewards.value_reward import normalize_value


def test_units_and_model_names_are_normalized() -> None:
    assert normalize_value("0.25 m") == normalize_value("25 cm")
    assert normalize_value("MODEL-X") == normalize_value("Model X")


def test_configured_color_aliases_are_normalized() -> None:
    assert normalize_value("navy blue") == normalize_value("深蓝") == "navy"

