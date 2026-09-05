"""Shared model-visible prompt and fixed four-field audit protocol."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROMPT = """你是一名电商商品图文一致性质检员。请检查完整商品页中的标题、品类、颜色、材质以及 main、detail:1、detail:2 三张图片。

输入包含：

- title：商品标题
- category：商品品类
- color：商品颜色，可能为 null
- material：商品材质，可能为 null
- main：主图
- detail:1：细节图 1
- detail:2：细节图 2

color 或 material 为 null 时不构成错误，不得根据图片补写缺失值。每个样本最多只有一种错误。

可输出的类型：

1. pass：文本和三张图片一致。
2. duplicate_detail_image：detail:1 与 detail:2 完全相同或近重复。
3. image_quality：main、detail:1 或 detail:2 存在明显模糊、遮挡或低分辨率。
   issue_subtype 必须为 blur、occlusion 或 low_resolution。
4. wrong_image：main 或任意 detail 图片属于其他商品。
5. category_mismatch：category 与图片中的实际品类不一致。
6. color_mismatch：非空 color 与图片中可可靠判断的颜色不一致。
7. material_mismatch：非空 material 与图片中可可靠判断的材质族不一致。
8. title_mismatch：title 描述的商品身份、型号、用途或关键内容与图片不一致。

如果 title 为了同步修改颜色或材质而发生相应变化，只输出 color_mismatch 或 material_mismatch，不额外输出 title_mismatch。构造反例时其他文本可能同步修改，但每次只能输出一个目标错误。

决策规则：

- pass 的 decision 为 pass。
- 其余七类的 decision 均为 reject。

证据规则：

- pass、duplicate_detail_image、category_mismatch、color_mismatch、material_mismatch、title_mismatch：evidence 必须为 null。
- image_quality：evidence 必须是受损图片的单个 image_ref 字符串。
- wrong_image：evidence 必须是错误图片的单个 image_ref 字符串。
- image_ref 只能是 main、detail:1 或 detail:2。

只能输出合法 JSON，不得输出 Markdown 或额外内容。decision 只能使用小写字符串 pass 或 reject；violation_type 必须逐字使用上述小写枚举，pass 必须保持小写。固定字段结构如下；尖括号内容是取值约束，不得原样输出：

{ "decision": <"pass" 或 "reject">,
  "violation_type": <上述八类之一的 JSON 字符串>,
  "issue_subtype": <"blur"、"occlusion"、"low_resolution" 或 null>,
  "evidence": <null 或单个合法 image_ref 字符串> }"""

OUTPUT_KEYS = ("decision", "violation_type", "issue_subtype", "evidence")
VIOLATION_TYPES = {
    "pass",
    "duplicate_detail_image",
    "image_quality",
    "wrong_image",
    "category_mismatch",
    "color_mismatch",
    "material_mismatch",
    "title_mismatch",
}
QUALITY_SUBTYPES = {"blur", "occlusion", "low_resolution"}
IMAGE_EVIDENCE_TYPES = {"image_quality", "wrong_image"}
IMAGE_REFS = {"main", "detail:1", "detail:2"}
FORBIDDEN_MODEL_TEXT_KEYS = {
    "sample_id",
    "source_product_ids",
    "source_image_ids",
    "derived_image_id",
    "lineage",
    "transform",
    "transform_params",
    "changed_fields",
    "reward_model",
    "ground_truth",
    "dataset_stage",
    "split",
}


def _display_value(value: object) -> str:
    return "N/A" if value is None else str(value)


def product_prompt(
    title: object,
    category: object,
    color: object,
    material: object,
    *,
    image_placeholders: bool,
) -> str:
    lines = []
    if image_placeholders:
        lines.extend(
            [
                "main: <image>",
                "detail:1: <image>",
                "detail:2: <image>",
            ]
        )
    lines.extend(
        [
            f"title: {_display_value(title)}",
            f"category: {_display_value(category)}",
            f"color: {_display_value(color)}",
            f"material: {_display_value(material)}",
        ]
    )
    lines.extend(["", PROMPT])
    return "\n".join(lines)


def prompt_with_image_token(
    title: object,
    category: object,
    color: object,
    material: object,
) -> str:
    return product_prompt(
        title,
        category,
        color,
        material,
        image_placeholders=True,
    )


def validate_product_prompt(text: object, *, image_placeholders: int) -> None:
    if not isinstance(text, str):
        raise ValueError("product prompt must be text")
    required = ("title:", "category:", "color:", "material:")
    if not all(any(line.startswith(prefix) for line in text.splitlines()) for prefix in required):
        raise ValueError("product prompt must expose title/category/color/material")
    if text.count("<image>") != image_placeholders:
        raise ValueError(f"product prompt must contain {image_placeholders} image placeholders")
    if not text.endswith(PROMPT):
        raise ValueError("product prompt must end with the canonical audit prompt")


def structured_prompt(image_paths: list[str], text: str) -> list[dict[str, Any]]:
    if len(image_paths) != 3:
        raise ValueError("model prompt requires main and two detail images")
    validate_product_prompt(text, image_placeholders=0)
    role_labels = ("main: ", "\ndetail:1: ", "\ndetail:2: ")
    content: list[dict[str, Any]] = []
    for label, path in zip(role_labels, image_paths):
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": str(Path(path))})
    content.append({"type": "text", "text": f"\n{text}"})
    return [{"role": "user", "content": content}]


def assert_model_text_is_sanitized(messages: object) -> None:
    leaked: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            leaked.update(str(key) for key in value if key in FORBIDDEN_MODEL_TEXT_KEYS)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            for key in FORBIDDEN_MODEL_TEXT_KEYS:
                if re.search(rf"(?m)^\\s*{re.escape(key)}\\s*:", value):
                    leaked.add(key)

    visit(messages)
    if leaked:
        raise ValueError(f"model-visible prompt contains internal metadata keys: {sorted(leaked)}")


def validate_prediction_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"prediction must contain exactly {OUTPUT_KEYS}")
    violation_type = value.get("violation_type")
    if violation_type not in VIOLATION_TYPES:
        raise ValueError(f"invalid violation_type: {violation_type!r}")
    expected_decision = "pass" if violation_type == "pass" else "reject"
    if value.get("decision") != expected_decision:
        raise ValueError(f"{violation_type} requires decision={expected_decision}")

    issue_subtype = value.get("issue_subtype")
    if violation_type == "image_quality":
        if issue_subtype not in QUALITY_SUBTYPES:
            raise ValueError(f"invalid image_quality issue_subtype: {issue_subtype!r}")
    elif issue_subtype is not None:
        raise ValueError(f"{violation_type} requires issue_subtype=null")

    evidence = value.get("evidence")
    if violation_type in IMAGE_EVIDENCE_TYPES:
        if not isinstance(evidence, str) or evidence not in IMAGE_REFS:
            raise ValueError(f"{violation_type} evidence requires one valid image_ref string")
    elif evidence is not None:
        raise ValueError(f"{violation_type} requires evidence=null")
    return {key: value[key] for key in OUTPUT_KEYS}


def target_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    violation_type = sample.get("violation_type")
    issue_subtype = sample.get("issue_subtype") if violation_type == "image_quality" else None
    evidence = None
    if violation_type in IMAGE_EVIDENCE_TYPES:
        evidence = sample.get("target_image_ref")
        if evidence not in IMAGE_REFS:
            candidate = sample.get("evidence")
            if isinstance(candidate, str) and candidate in IMAGE_REFS:
                evidence = candidate
        if evidence not in IMAGE_REFS:
            raise ValueError(f"{violation_type} requires exactly one valid image_ref")
    return validate_prediction_dict(
        {
            "decision": "pass" if violation_type == "pass" else "reject",
            "violation_type": violation_type,
            "issue_subtype": issue_subtype,
            "evidence": evidence,
        }
    )
