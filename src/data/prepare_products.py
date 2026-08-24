from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.common import load_yaml, stable_hash
from src.data.split_manifest import stable_split_for, write_split_manifests

CATEGORIES = ("bottle", "headphones", "backpack", "shoe")
COLORS = ("navy", "red", "green", "orange", "purple", "black")
MATERIALS = ("steel", "plastic", "canvas", "leather")


def _draw_product(path: Path, category: str, color: str, model: str, variant: int) -> None:
    palette = {
        "navy": "#183153",
        "red": "#b42318",
        "green": "#177245",
        "orange": "#e26d21",
        "purple": "#6938a8",
        "black": "#252525",
    }
    image = Image.new("RGB", (520, 420), "#f4f1ea")
    draw = ImageDraw.Draw(image)
    fill = palette[color]
    offset = variant * 5
    if category == "bottle":
        draw.rounded_rectangle((175 + offset, 75, 345 + offset, 350), radius=28, fill=fill)
        draw.rectangle((215 + offset, 45, 305 + offset, 95), fill="#333333")
    elif category == "headphones":
        draw.arc((115, 45 + offset, 405, 325 + offset), 190, 350, fill=fill, width=35)
        draw.rounded_rectangle((90, 190, 175, 345), radius=25, fill=fill)
        draw.rounded_rectangle((345, 190, 430, 345), radius=25, fill=fill)
    elif category == "backpack":
        draw.rounded_rectangle((125, 85 + offset, 395, 355 + offset), radius=45, fill=fill)
        draw.arc((190, 30, 330, 145), 180, 360, fill="#444444", width=18)
        draw.rectangle((165, 235 + offset, 355, 320 + offset), outline="#f4f1ea", width=8)
    else:
        draw.polygon([(75, 270), (205, 180 - offset), (300, 270), (445, 305), (430, 355), (120, 355)], fill=fill)
        draw.line((120, 320, 430, 320), fill="#f4f1ea", width=8)
    font = ImageFont.load_default()
    draw.rounded_rectangle((15, 15, 145, 45), radius=8, fill="white")
    draw.text((25, 23), model, fill="#222222", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def build_products(config: dict) -> list[dict]:
    if config.get("source_mode") != "synthetic_demo":
        raise ValueError("This portfolio build currently supports source_mode=synthetic_demo only")
    seed = int(config["seed"])
    rng = random.Random(seed)
    raw_dir = Path(config["paths"]["raw_images"])
    products: list[dict] = []
    for index in range(int(config["num_products"])):
        product_id = f"product_{index:05d}"
        category = CATEGORIES[index % len(CATEGORIES)]
        color = COLORS[index % len(COLORS)]
        material = MATERIALS[index % len(MATERIALS)]
        model = f"{category[:2].upper()}-{1000 + index}"
        brand = f"DemoBrand-{chr(65 + index % 8)}"
        image_paths = []
        image_ids = []
        image_assets = []
        for variant in range(3):
            image_id = f"synthetic_{product_id}_view_{variant}"
            path = raw_dir / "by_id" / image_id / "image.jpg"
            _draw_product(path, category, color, model, variant)
            image_paths.append(path.as_posix())
            image_ids.append(image_id)
            image_assets.append(
                {
                    "source_image_id": image_id,
                    "path": path.as_posix(),
                    "role": "main" if variant == 0 else "gallery",
                    "source": "synthetic_demo",
                }
            )
        products.append(
            {
                "product_id": product_id,
                "family_id": f"family_{index:05d}",
                "images": image_paths,
                "image_ids": image_ids,
                "image_assets": image_assets,
                "title": f"{brand} {color.title()} {category.title()} {model}",
                "category": category,
                "brand": brand,
                "attributes": {
                    "color": color,
                    "material": material,
                    "model": model,
                    "dimensions": f"{20 + index % 10} cm",
                },
                "required_fields": ["color", "material", "model"],
                "source": "synthetic_demo",
                "license": "generated-for-this-project",
                "source_record_id": product_id,
                "source_archive_sha256": None,
                "generation_seed": rng.randrange(2**31),
            }
        )
    return products


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic synthetic product pool")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if config.get("source_mode") == "abo_rendered_audit":
        from src.data.prepare_abo import prepare_abo_products

        products = prepare_abo_products(config)
        print(f"prepared {len(products)} selectively downloaded ABO products")
        return
    products = build_products(config)
    for product in products:
        product["source_component_id"] = product["family_id"]
        product["split"] = stable_split_for(
            product["source_component_id"],
            int(config["seed"]),
            config["split_ratios"],
        )
    targets = write_split_manifests(config, "products", products)
    print(
        f"wrote {len(products)} split products to {', '.join(str(path) for path in targets.values())} "
        f"(content_hash={stable_hash(products)[:12]})"
    )


if __name__ == "__main__":
    main()
