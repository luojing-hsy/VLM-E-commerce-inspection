from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal["pass", "review", "reject"]
ViolationType = Literal[
    "PASS",
    "PRODUCT_MISMATCH",
    "ATTRIBUTE_CONFLICT",
    "TEXT_LABEL_CONFLICT",
    "MISSING_REQUIRED_FIELD",
    "IMAGE_QUALITY",
    "IRRELEVANT_IMAGE",
    "DUPLICATE_IMAGE",
]
RegionType = Literal["bbox", "image_ref", "image_pair", "missing_field"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    region_type: RegionType
    image_ref: str | None = None
    image_refs: list[str] | None = None
    bbox_norm: tuple[int, int, int, int] | None = None
    field: str | None = None
    value: str | None = None
    evidence_source: str | None = None
    source_field: str | None = None

    @model_validator(mode="after")
    def validate_region(self) -> "Evidence":
        if self.region_type == "bbox":
            if self.image_ref is None or self.bbox_norm is None:
                raise ValueError("bbox evidence requires image_ref and bbox_norm")
            x1, y1, x2, y2 = self.bbox_norm
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError("bbox_norm must be valid xyxy coordinates in [0, 1000]")
        elif self.region_type == "image_pair":
            if self.image_refs is None or len(self.image_refs) != 2:
                raise ValueError("image_pair evidence requires exactly two image_refs")
        elif self.region_type == "image_ref" and self.image_ref is None:
            raise ValueError("image_ref evidence requires image_ref")
        elif self.region_type == "missing_field" and not self.field:
            raise ValueError("missing_field evidence requires field")
        return self


class AuditPrediction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["1.0"] = "1.0"
    decision: Decision
    violation_type: ViolationType | None
    field: str | None = None
    listed_value: str | None = None
    observed_value: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_protocol(self) -> "AuditPrediction":
        if self.decision == "pass":
            if self.violation_type not in (None, "PASS"):
                raise ValueError("pass decision cannot contain a violation type")
            if any(value is not None for value in (self.field, self.listed_value, self.observed_value)):
                raise ValueError("PASS fields must be null")
            if self.evidence:
                raise ValueError("PASS evidence must be empty")
            self.violation_type = "PASS"
        elif self.violation_type in (None, "PASS"):
            raise ValueError("review/reject requires a non-PASS violation type")
        return self

