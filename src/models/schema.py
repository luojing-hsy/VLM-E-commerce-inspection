from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from src.models.audit_protocol import validate_prediction_dict


Decision = Literal["pass", "reject"]
ViolationType = Literal[
    "pass",
    "duplicate_detail_image",
    "image_quality",
    "wrong_image",
    "category_mismatch",
    "color_mismatch",
    "material_mismatch",
    "title_mismatch",
]
IssueSubtype = Literal["blur", "occlusion", "low_resolution"]
ImageRef = Literal["main", "detail:1", "detail:2"]


class AuditPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    violation_type: ViolationType
    issue_subtype: IssueSubtype | None
    evidence: ImageRef | None

    @model_validator(mode="after")
    def validate_protocol(self) -> "AuditPrediction":
        validate_prediction_dict(self.model_dump(mode="python"))
        return self
