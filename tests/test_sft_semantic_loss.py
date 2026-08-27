from src.training.sft_semantic import semantic_completion_mask


class CharacterTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def test_sft_semantic_mask_includes_values_but_excludes_json_shell() -> None:
    text = (
        '{"decision":"reject","violation_type":"image_quality",'
        '"issue_subtype":"blur","evidence":"detail:1"}'
    )
    mask = semantic_completion_mask(CharacterTokenizer(), [ord(char) for char in text])

    for value in ("reject", "image_quality", "blur", "detail:1"):
        start = text.index(value)
        assert all(mask[start : start + len(value)])
    for shell in ('"decision"', '"violation_type"', '"issue_subtype"', '"evidence"'):
        start = text.index(shell)
        assert not any(mask[start : start + len(shell)])
    for quoted_value in ('"reject"', '"image_quality"', '"blur"', '"detail:1"'):
        start = text.index(quoted_value)
        assert not mask[start]
        assert not mask[start + len(quoted_value) - 1]

    assert any(mask)
    assert sum(mask) < len(mask)
