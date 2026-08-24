from __future__ import annotations

import itertools


def bbox_iou(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _item_score(expected: dict, predicted: dict) -> float:
    if expected.get("role") != predicted.get("role") or expected.get("region_type") != predicted.get("region_type"):
        return 0.0
    region_type = expected["region_type"]
    if region_type == "bbox":
        if predicted.get("image_ref") != expected.get("image_ref") or not predicted.get("bbox_norm"):
            return 0.0
        return bbox_iou(expected["bbox_norm"], predicted["bbox_norm"])
    if region_type == "image_ref":
        return float(expected.get("image_ref") == predicted.get("image_ref"))
    if region_type == "image_pair":
        return float(set(expected.get("image_refs", [])) == set(predicted.get("image_refs", [])))
    if region_type == "missing_field":
        return float(expected.get("field") == predicted.get("field"))
    return 0.0


def evidence_reward(sample: dict, prediction: dict) -> float:
    expected = sample.get("evidence", [])
    predicted = prediction.get("evidence") or []
    if not expected:
        return 0.0
    if not predicted:
        return 0.0
    # Evidence sets are tiny (at most two in V1), so an exact assignment is simpler
    # and more portable than adding SciPy solely for Hungarian matching.
    padded = list(predicted[: len(expected)])
    while len(padded) < len(expected):
        padded.append({})
    best = 0.0
    for permutation in itertools.permutations(padded):
        score = sum(_item_score(target, candidate) for target, candidate in zip(expected, permutation)) / len(expected)
        best = max(best, score)
    return best

