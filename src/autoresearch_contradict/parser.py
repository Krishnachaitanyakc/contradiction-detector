"""Parse results.tsv and categorize experiments by change type."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

CHANGE_TYPE_PATTERNS = {
    "lr_change": [r"learning.?rate", r"\blr\b", r"step.?size"],
    "architecture_change": [
        r"batch.?norm", r"layer.?norm", r"skip.?connect",
        r"resid", r"attention", r"hidden.?size", r"num.?layers",
        r"depth", r"width",
    ],
    "regularization_change": [
        r"dropout", r"regulariz", r"weight.?decay",
        r"l1\b", r"l2\b", r"label.?smooth",
    ],
    "optimizer_change": [
        r"adam\b", r"sgd\b", r"optim", r"rmsprop",
        r"adagrad", r"momentum",
    ],
    "data_augmentation_change": [
        r"augment", r"random.?crop", r"flip", r"mixup",
        r"cutout", r"cutmix",
    ],
}


@dataclass
class ExperimentRecord:
    """A parsed experiment record with change type categorization."""

    commit: str
    metric_value: float
    memory_gb: float
    status: str
    description: str
    change_type: str


class ExperimentParser:
    """Parse and categorize experiment records."""

    def parse_tsv(self, tsv_content: str) -> List[ExperimentRecord]:
        """Parse TSV string into categorized experiment records."""
        lines = tsv_content.strip().split("\n")
        if len(lines) <= 1:
            return []

        records = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            description = parts[4].strip()
            change_type = self._categorize(description)
            records.append(
                ExperimentRecord(
                    commit=parts[0].strip(),
                    metric_value=float(parts[1].strip()),
                    memory_gb=float(parts[2].strip()),
                    status=parts[3].strip(),
                    description=description,
                    change_type=change_type,
                )
            )
        return records

    def parse_tsv_file(self, filepath: str) -> List[ExperimentRecord]:
        """Parse a TSV file into experiment records."""
        with open(filepath, "r") as f:
            return self.parse_tsv(f.read())

    def _categorize(self, description: str) -> str:
        """Categorize an experiment by its description."""
        desc_lower = description.lower()
        for change_type, patterns in CHANGE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, desc_lower):
                    return change_type
        return "other"

    @staticmethod
    def determine_direction(metric: float, baseline: float) -> str:
        """Determine if metric is positive, negative, or neutral vs baseline."""
        if metric > baseline:
            return "positive"
        elif metric < baseline:
            return "negative"
        else:
            return "neutral"
