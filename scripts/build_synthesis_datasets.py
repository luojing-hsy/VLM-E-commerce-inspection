"""Build three derived JSONL samples for every high-resolution split product.

The output keeps the compact ``all_product.jsonl`` shape.  Image replacements
are path-only replacements except for image_quality, whose transformed image is
materialized below the corresponding synthesis directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
RAW_JSONL = ROOT / "data" / "all_product.jsonl"
SPLITS = {
    "sft_train": ("data/highres_split/SFT/train/manifest.jsonl", "data/sft_synthesis/train.jsonl"),
    "sft_valid": ("data/highres_split/SFT/valid/manifest.jsonl", "data/sft_synthesis/valid.jsonl"),
    "joint_train": ("data/highres_split/GRPO+OPD/train/manifest.jsonl", "data/joint_synthesis/train.jsonl"),
    "joint_valid": ("data/highres_split/GRPO+OPD/valid/manifest.jsonl", "data/joint_synthesis/valid.jsonl"),
    "test": ("data/highres_split/test/manifest.jsonl", "data/test_synthesis/test.jsonl"),
}

# Counts are sample counts.  Every source product receives exactly three rows.
QUOTAS = {
    "sft_train": {"pass": 840, "duplicate_detail_image": 840, "image_quality": 840,
                  "category_mismatch": 420, "wrong_image": 420, "color_mismatch": 420,
                  "title_mismatch": 210, "material_mismatch": 210},
    "sft_valid": {"pass": 72, "duplicate_detail_image": 72, "image_quality": 72,
                  "category_mismatch": 36, "wrong_image": 36, "color_mismatch": 36,
                  "title_mismatch": 18, "material_mismatch": 18},
    "joint_train": {"pass": 600, "duplicate_detail_image": 600, "image_quality": 600,
                     "category_mismatch": 1115, "wrong_image": 1115, "color_mismatch": 770,
                     "title_mismatch": 763, "material_mismatch": 437},
    "joint_valid": {"pass": 54, "duplicate_detail_image": 54, "image_quality": 54,
                     "category_mismatch": 137, "wrong_image": 68, "color_mismatch": 65,
                     "title_mismatch": 77, "material_mismatch": 31},
    "test": {"pass": 143, "duplicate_detail_image": 143, "image_quality": 143,
             "category_mismatch": 86, "wrong_image": 85, "color_mismatch": 86,
             "title_mismatch": 117, "material_mismatch": 55},
}

EASY = ("pass", "duplicate_detail_image", "image_quality")
MEDIUM = ("category_mismatch", "wrong_image", "color_mismatch")
HARD = ("title_mismatch", "material_mismatch")
ALL_LABELS = EASY + MEDIUM + HARD



class TitleMutationError(ValueError):
    """Raised when a title contains an attribute mention that cannot be safely rewritten."""


_ATTRIBUTE_FIELDS = {"category", "color", "material"}
_ATTRIBUTE_ALIAS_GROUPS = {
    "color": (("gray", "grey"), ("navy", "navy blue")),
    "material": (("aluminum", "aluminium"), ("stainless steel", "stainless-steel")),
    "category": (),
}
_TITLE_TOKEN = re.compile(r"[A-Za-z0-9]+")
_TITLE_SEPARATOR = r"(?:[\s_./\\,&:;()+\[\]–—-]+|\s+and\s+)"

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_int(seed: int, namespace: str, value: str) -> int:
    return int(hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()[:16], 16)


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", norm(value)))


def _lexical_distance(left: object, right: object) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    overlap = left_tokens & right_tokens
    distance = 1.0 - len(overlap) / len(union)
    if norm(left)[:1] != norm(right)[:1]:
        distance += 0.05
    return distance


def _color_family(value: object) -> str | None:
    text = norm(value)
    families = {
        "neutral": ("black", "white", "gray", "grey", "silver", "clear"),
        "warm": ("red", "orange", "yellow", "pink", "coral", "gold"),
        "cool": ("blue", "navy", "teal", "cyan", "turquoise"),
        "green": ("green", "olive", "lime", "mint"),
        "purple": ("purple", "violet", "lavender", "plum"),
        "earth": ("brown", "tan", "beige", "khaki", "cream", "ivory"),
    }
    return next((family for family, words in families.items() if any(word in text for word in words)), None)


def _material_family(value: object) -> str | None:
    text = norm(value)
    families = {
        "metal": ("metal", "steel", "aluminum", "aluminium", "brass", "iron", "zinc", "alloy", "chrome", "copper", "nickel", "silver", "gold"),
        "wood": ("wood", "bamboo", "walnut", "oak", "pine", "hardwood", "maple", "birch", "teak", "plywood", "rattan", "wicker", "mdf", "fibreboard"),
        "leather": ("leather", "suede"),
        "plastic": ("plastic", "poly", "pvc", "pla", "nylon", "silicone", "rubber", "nitrile", "tpu", "resin", "abs", "epp", "petg", "polycarbonate", "polyurethane", "melamine"),
        "glass": ("glass", "crystal"),
        "fabric": ("fabric", "textile", "cotton", "polyester", "canvas", "linen", "neoprene", "foam", "jute", "wool", "microfiber", "felt", "velvet", "denim", "plush", "fur", "sisal"),
        "ceramic": ("ceramic", "stoneware", "porcelain", "stone", "earthenware"),
        "paper": ("paper", "cardboard"),
    }
    return next((family for family, words in families.items() if any(word in text for word in words)), None)


def _value_distance(field: str, original: object, candidate: object) -> float:
    family_fn = _color_family if field == "color" else _material_family
    original_family, candidate_family = family_fn(original), family_fn(candidate)
    distance = _lexical_distance(original, candidate)
    if original_family and candidate_family and original_family != candidate_family:
        distance += 2.0
    elif original_family and candidate_family and original_family == candidate_family:
        distance *= 0.2
    return distance

def _title_tokens(value: object) -> list[str]:
    return _TITLE_TOKEN.findall(str(value or "").casefold())


def _attribute_aliases(field: str, value: object) -> list[str]:
    if field not in _ATTRIBUTE_FIELDS:
        raise ValueError(f"unsupported title attribute field: {field}")
    value_text = str(value or "").strip()
    aliases = {value_text} if value_text else set()
    value_tokens = " ".join(_title_tokens(value_text))
    for group in _ATTRIBUTE_ALIAS_GROUPS[field]:
        if value_tokens in {" ".join(_title_tokens(alias)) for alias in group}:
            aliases.update(group)
    return sorted(aliases, key=lambda alias: (-len(_title_tokens(alias)), -len(alias), alias.casefold()))


def _phrase_pattern(value: object) -> re.Pattern[str] | None:
    tokens = _title_tokens(value)
    if not tokens:
        return None
    phrase = _TITLE_SEPARATOR.join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![A-Za-z0-9]){phrase}(?![A-Za-z0-9])", flags=re.IGNORECASE)


def _attribute_pattern(field: str, value: object) -> re.Pattern[str] | None:
    patterns = []
    for alias in _attribute_aliases(field, value):
        pattern = _phrase_pattern(alias)
        if pattern is not None:
            patterns.append(pattern.pattern)
    if not patterns:
        return None
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns), flags=re.IGNORECASE)


def _has_unordered_attribute_mention(title: str, field: str, value: object) -> bool:
    """Detect a likely reordered multi-token mention and fail closed."""

    del field
    tokens = _title_tokens(value)
    if len(tokens) < 2:
        return False
    title_tokens = _title_tokens(title)
    required = set(tokens)
    max_window = len(tokens) + 3
    for start in range(len(title_tokens)):
        for end in range(start + len(tokens), min(len(title_tokens), start + max_window) + 1):
            if required.issubset(title_tokens[start:end]):
                return True
    return False


def replace_text(text: str, old: object, new: object) -> str:
    """Replace every boundary-aware occurrence of old in text."""

    pattern = _phrase_pattern(old)
    if pattern is None or norm(old) == norm(new):
        return text
    return pattern.sub(str(new), text)


def mutate_title_attribute(title: str, field: str, old: object, new: object) -> tuple[str, dict]:
    """Synchronize all safe title mentions of one structured attribute.

    A title without an explicit mention is valid and recorded as such. A
    reordered multi-token mention is ambiguous and raises instead of being
    silently left unchanged.
    """

    if field not in _ATTRIBUTE_FIELDS:
        raise ValueError(f"unsupported title attribute field: {field}")
    if old in (None, "") or new in (None, ""):
        raise TitleMutationError(f"{field} title mutation requires non-empty old and new values")
    pattern = _attribute_pattern(field, old)
    matches = list(pattern.finditer(title)) if pattern is not None else []
    if not matches:
        if _has_unordered_attribute_mention(title, field, old):
            raise TitleMutationError(
                f"ambiguous reordered {field} mention for {old!r} in title {title!r}"
            )
        return title, {
            "field": field,
            "original": str(old),
            "modified": str(new),
            "policy": "no_mention",
            "source_mentions": 0,
            "replaced_mentions": 0,
            "matched_texts": [],
        }

    mutated, replaced_count = pattern.subn(str(new), title)
    if replaced_count != len(matches):
        raise TitleMutationError(
            f"{field} title replacement count changed during mutation: "
            f"matched={len(matches)} replaced={replaced_count}"
        )
    return mutated, {
        "field": field,
        "original": str(old),
        "modified": str(new),
        "policy": "replace_all",
        "source_mentions": len(matches),
        "replaced_mentions": replaced_count,
        "matched_texts": [match.group(0) for match in matches],
    }


def attribute_title_eligible(title: str, field: str, value: object) -> bool:
    """Return whether a title can safely participate in an attribute mutation."""

    if value in (None, ""):
        return False
    pattern = _attribute_pattern(field, value)
    if pattern is not None and pattern.search(title):
        return True
    return not _has_unordered_attribute_mention(title, field, value)



def source_path(split_manifest: Path, name: str) -> str:
    # Keep paths relative to the repository, as in all_product.jsonl.
    return (split_manifest.parent / name).relative_to(ROOT).as_posix()


def asset_paths(split_manifest: Path, manifest_row: dict) -> dict[str, str]:
    return {
        str(asset["role"]): source_path(split_manifest, str(asset["high_resolution_path"]))
        for asset in manifest_row["highres_images"]
    }


def make_base(raw: dict, paths: dict[str, str]) -> dict:
    return {
        "product_id": str(raw["product_id"]),
        "title": raw["title"],
        "category": raw["category"],
        "color": raw.get("color"),
        "material": raw.get("material"),
        "images": {
            "main": {"image_id": paths["main"]},
            "detail": [
                {"image_id": paths["detail:1"]},
                {"image_id": paths["detail:2"]},
            ],
        },
    }


def same_attributes(a: dict, b: dict) -> bool:
    for field in ("color", "material"):
        av, bv = a.get(field), b.get(field)
        if av not in (None, "") and bv not in (None, "") and norm(av) != norm(bv):
            return False
    return True


def candidate_donors(rows: list[dict], raw_by_id: dict[str, dict], label: str) -> dict[str, list[dict]]:
    by_category: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    result: dict[str, list[dict]] = {}
    for row in rows:
        pid = str(row["product_id"])
        category_rows = [x for x in by_category[row["category"]] if x["product_id"] != pid]
        if label == "title_mismatch":
            result[pid] = [x for x in category_rows if same_attributes(raw_by_id[pid], raw_by_id[str(x["product_id"])])]
        elif label == "wrong_image":
            compatible = [x for x in category_rows if same_attributes(raw_by_id[pid], raw_by_id[str(x["product_id"])])]
            result[pid] = compatible or category_rows
        else:
            result[pid] = category_rows
    return result


def choose_value(rows: list[dict], raw_by_id: dict[str, dict], pid: str, field: str, used: set[str], seed: int) -> object:
    original = raw_by_id[pid].get(field)
    candidates = [raw_by_id[str(row["product_id"])].get(field) for row in rows]
    candidates = [value for value in candidates if value not in (None, "") and norm(value) != norm(original)]
    candidates = list(dict.fromkeys(str(value) for value in candidates))
    if not candidates:
        raise RuntimeError(f"no distinct {field} value in split for {pid}")
    ordered = sorted(candidates, key=lambda value: (_value_distance(field, original, value), stable_int(seed, f"{field}:{pid}", value)), reverse=True)
    for value in ordered:
        if norm(value) not in used:
            used.add(norm(value))
            return value
    return ordered[0]


def choose_material(rows: list[dict], raw_by_id: dict[str, dict], pid: str, used: set[str], seed: int) -> object:
    return choose_value(rows, raw_by_id, pid, "material", used, seed)


def quality_image(source: Path, destination: Path, subtype: str, seed: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if subtype == "blur":
        output = image.filter(ImageFilter.GaussianBlur(radius=10))
    elif subtype == "low_resolution":
        small_width = max(32, image.width // 8)
        small_height = max(32, image.height // 8)
        output = image.resize((small_width, small_height), Image.Resampling.BILINEAR)
        output = output.resize(image.size, Image.Resampling.NEAREST)
    else:
        output = image.copy()
        rng = random.Random(seed)
        draw = ImageDraw.Draw(output)
        width, height = output.size
        box_width = max(20, round(width * 0.28))
        box_height = max(20, round(height * 0.28))
        x = rng.randint(0, max(0, width - box_width))
        y = rng.randint(0, max(0, height - box_height))
        draw.rectangle((x, y, x + box_width, y + box_height), fill=(20, 20, 20))
    output.save(destination, quality=95)


def _assign_labels_flow(rows: list[dict], raw_by_id: dict[str, dict], quota: dict[str, int], seed: int) -> dict[str, list[str]]:
    """Allocate labels globally with capacity three per product and no duplicate label."""
    product_ids = [str(row["product_id"]) for row in rows]
    donors = {label: candidate_donors(rows, raw_by_id, label) for label in ("title_mismatch", "wrong_image")}
    categories = {str(row["category"]) for row in rows}

    def eligible(label: str, pid: str) -> bool:
        raw = raw_by_id[pid]
        if label == "color_mismatch":
            return (
                raw.get("color") not in (None, "")
                and attribute_title_eligible(str(raw["title"]), "color", raw["color"])
            )
        if label == "material_mismatch":
            return (
                raw.get("material") not in (None, "")
                and attribute_title_eligible(str(raw["title"]), "material", raw["material"])
            )
        if label in donors:
            return bool(donors[label][pid])
        if label == "category_mismatch":
            return len(categories) > 1 and attribute_title_eligible(str(raw["title"]), "category", raw["category"])
        return True

    labels = list(quota)
    source = 0
    label_start = 1
    product_start = label_start + len(labels)
    sink = product_start + len(product_ids)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, capacity: int) -> None:
        graph[left].append([right, capacity, len(graph[right])])
        graph[right].append([left, 0, len(graph[left]) - 1])

    for label_index, label in enumerate(labels):
        add_edge(source, label_start + label_index, quota[label])
        for product_index, pid in enumerate(product_ids):
            if eligible(label, pid):
                add_edge(label_start + label_index, product_start + product_index, 1)
    for product_index in range(len(product_ids)):
        add_edge(product_start + product_index, sink, 3)

    flow = 0
    while True:
        parent = [None] * len(graph)
        parent[source] = (-1, -1)
        queue = [source]
        for node in queue:
            for edge_index, edge in enumerate(graph[node]):
                if edge[1] > 0 and parent[edge[0]] is None:
                    parent[edge[0]] = (node, edge_index)
                    queue.append(edge[0])
        if parent[sink] is None:
            break
        node = sink
        while node != source:
            previous, edge_index = parent[node]
            edge = graph[previous][edge_index]
            edge[1] -= 1
            graph[node][edge[2]][1] += 1
            node = previous
        flow += 1

    requested = sum(quota.values())
    if flow != requested:
        raise RuntimeError(f"cannot allocate exact label quotas: requested {requested}, allocated {flow}")
    assignments = {pid: [] for pid in product_ids}
    for label_index, label in enumerate(labels):
        node = label_start + label_index
        for edge in graph[node]:
            product_node, capacity, _ = edge
            if product_start <= product_node < sink and capacity == 0:
                assignments[product_ids[product_node - product_start]].append(label)
    for pid in assignments:
        assignments[pid].sort(key=lambda label: (ALL_LABELS.index(label), stable_int(seed, "slot", pid + label)))
    if any(len(values) != 3 or len(values) != len(set(values)) for values in assignments.values()):
        raise AssertionError("every product must have exactly three distinct labels")
    return assignments


def assign_labels(rows: list[dict], raw_by_id: dict[str, dict], quota: dict[str, int], seed: int) -> dict[str, list[str]]:
    return _assign_labels_flow(rows, raw_by_id, quota, seed)

ATTRIBUTE_LABEL_FIELDS = {
    "category_mismatch": "category",
    "color_mismatch": "color",
    "material_mismatch": "material",
}


def adjust_quota_for_title_eligibility(
    rows: list[dict],
    raw_by_id: dict[str, dict],
    quota: dict[str, int],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Move unavailable attribute quota to pass instead of forcing unsafe rows."""

    effective = dict(quota)
    adjustments: dict[str, dict[str, int]] = {}
    for label, field in ATTRIBUTE_LABEL_FIELDS.items():
        capacity = sum(
            attribute_title_eligible(
                str(raw_by_id[str(row["product_id"])]["title"]),
                field,
                raw_by_id[str(row["product_id"])].get(field),
            )
            for row in rows
        )
        requested = int(effective.get(label, 0))
        if requested <= capacity:
            continue
        moved = requested - capacity
        effective[label] = capacity
        effective["pass"] = int(effective.get("pass", 0)) + moved
        adjustments[label] = {
            "requested": requested,
            "eligible_capacity": capacity,
            "moved_to_pass": moved,
        }
    if sum(effective.values()) != len(rows) * 3:
        raise RuntimeError(
            f"effective quotas must allocate three labels per product: "
            f"products={len(rows)} quota_total={sum(effective.values())}"
        )
    return effective, adjustments


def build_partition(name: str, raw_by_id: dict[str, dict], seed: int) -> tuple[list[dict], dict]:
    manifest_rel, output_rel = SPLITS[name]
    manifest_path = ROOT / manifest_rel
    manifest_rows = read_jsonl(manifest_path)
    rows = sorted(manifest_rows, key=lambda row: str(row["product_id"]))
    if len(rows) * 3 != sum(QUOTAS[name].values()):
        raise AssertionError(f"{name}: product/sample total mismatch")
    requested_quota = dict(QUOTAS[name])
    effective_quota, quota_adjustments = adjust_quota_for_title_eligibility(rows, raw_by_id, requested_quota)
    assignments = assign_labels(rows, raw_by_id, effective_quota, seed)
    donor_cache = {label: candidate_donors(rows, raw_by_id, label) for label in ("title_mismatch", "wrong_image")}
    output_path = ROOT / output_rel
    output_root = output_path.parent
    quality_root = output_root / "images"
    output_rows: list[dict] = []
    quality_count = 0
    used_values: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    donor_cursor: defaultdict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        pid = str(row["product_id"])
        raw = raw_by_id[pid]
        paths = asset_paths(manifest_path, row)
        base = make_base(raw, paths)
        labels = assignments[pid]
        for index, label in enumerate(labels, start=1):
            sample = deepcopy(base)
            sample_id = f"{pid}_{index:02d}"
            sample["product_id"] = sample_id
            sample.update({"source_product_id": pid, "dataset": name, "sample_index": index,
                           "difficulty": "easy" if label in EASY else "medium" if label in MEDIUM else "hard",
                           "violation_type": label})
            if label == "duplicate_detail_image":
                sample["images"]["detail"][1]["image_id"] = paths["detail:1"]
            elif label == "image_quality":
                target_role = ("main", "detail:1", "detail:2")[quality_count % 3]
                subtype = ("blur", "low_resolution", "occlusion")[quality_count % 3]
                original_path = Path(sample["images"]["main"]["image_id"] if target_role == "main" else sample["images"]["detail"][int(target_role[-1]) - 1]["image_id"])
                output_name = f"{original_path.stem}_image_quality{original_path.suffix}"
                destination = quality_root / output_name
                quality_image(ROOT / original_path, destination, subtype, stable_int(seed, "quality", sample_id))
                quality_path = destination.relative_to(ROOT).as_posix()
                if target_role == "main":
                    sample["images"]["main"]["image_id"] = quality_path
                else:
                    sample["images"]["detail"][int(target_role[-1]) - 1]["image_id"] = quality_path
                sample["issue_subtype"] = subtype
                sample["target_image_ref"] = target_role
                quality_count += 1
            elif label == "wrong_image":
                donors = donor_cache[label][pid]
                cursor_key = (pid, label)
                donor = donors[donor_cursor[cursor_key] % len(donors)]
                donor_cursor[cursor_key] += 1
                donor_paths = asset_paths(manifest_path, donor)
                detail_index = (index - 1) % 2
                sample["images"]["detail"][detail_index]["image_id"] = donor_paths["detail:1" if detail_index == 0 else "detail:2"]
                sample["target_image_ref"] = f"detail:{detail_index + 1}"
            elif label == "category_mismatch":
                categories = sorted({str(other["category"]) for other in rows if str(other["category"]) != str(raw["category"])})
                value = max(categories, key=lambda candidate: (_lexical_distance(raw.get("category"), candidate), stable_int(seed, "category", sample_id + candidate)))
                sample["category"] = value
                sample["title"], sample["title_audit"] = mutate_title_attribute(
                    sample["title"], "category", raw.get("category"), value
                )
                sample["changed_field"] = "category"
            elif label == "color_mismatch":
                value = choose_value(rows, raw_by_id, pid, "color", used_values[(pid, label)], seed)
                sample["color"] = value
                sample["title"], sample["title_audit"] = mutate_title_attribute(
                    sample["title"], "color", raw.get("color"), value
                )
                sample["changed_field"] = "color"
            elif label == "material_mismatch":
                value = choose_material(rows, raw_by_id, pid, used_values[(pid, label)], seed)
                sample["material"] = value
                sample["title"], sample["title_audit"] = mutate_title_attribute(
                    sample["title"], "material", raw.get("material"), value
                )
                sample["changed_field"] = "material"
            elif label == "title_mismatch":
                donors = donor_cache[label][pid]
                cursor_key = (pid, label)
                donor = donors[donor_cursor[cursor_key] % len(donors)]
                donor_cursor[cursor_key] += 1
                sample["title"] = raw_by_id[str(donor["product_id"])] ["title"]
            output_rows.append(sample)

    output_rows.sort(key=lambda row: row["product_id"])
    write_jsonl(output_path, output_rows)
    stats = {
        "dataset": name,
        "output": output_rel,
        "products": len(rows),
        "samples": len(output_rows),
        "samples_per_product": dict(Counter(Counter(row["source_product_id"] for row in output_rows).values())),
        "label_counts": dict(sorted(Counter(row["violation_type"] for row in output_rows).items())),
        "requested_label_counts": requested_quota,
        "effective_label_counts": effective_quota,
        "quota_adjustments": quota_adjustments,
        "quality_images_created": quality_count,
    }
    # Replace the non-serializable Counter-values view with a useful invariant.
    stats["products_with_three_samples"] = sum(1 for count in Counter(row["source_product_id"] for row in output_rows).values() if count == 3)
    (output_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    raw_rows = read_jsonl(RAW_JSONL)
    raw_by_id = {str(row["product_id"]): row for row in raw_rows}
    if len(raw_by_id) != 3986:
        raise ValueError(f"expected 3986 unique raw products, got {len(raw_by_id)}")
    all_rows: list[dict] = []
    reports = {}
    for offset, name in enumerate(SPLITS):
        rows, report = build_partition(name, raw_by_id, args.seed + offset)
        all_rows.extend(rows)
        reports[name] = report
    if len(all_rows) != 11958:
        raise AssertionError(f"expected 11958 samples, got {len(all_rows)}")
    report = {"total_samples": len(all_rows), "total_products": len(raw_rows), "partitions": reports}
    (ROOT / "data" / "synthesis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()










