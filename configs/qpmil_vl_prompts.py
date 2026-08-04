"""Task-name-keyed TITAN prompts for the 10-task, 27-class WSI stream.

This registry deliberately does not depend on task position.  Reversing the
dataset task order therefore reverses whole task prompt groups without
silently assigning prompts to the wrong global labels.
"""

from __future__ import annotations

import hashlib
import json
from typing import List, Mapping, Sequence, Tuple


TEMPLATES: Tuple[str, ...] = (
    "CLASSNAME.",
    "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.",
    "an image showing CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E.",
)


# One tuple per class, in the same local-label order as datasets/seq_wsi.py.
PROMPT_REGISTRY: Mapping[str, Tuple[Tuple[str, ...], ...]] = {
    "camelyon17": (
        (
            "negative sentinel lymph node",
            "sentinel lymph node without metastatic carcinoma",
            "no tumor cells in breast cancer sentinel lymph node",
            "benign lymph node tissue without metastasis",
        ),
        (
            "isolated tumor cells in sentinel lymph node",
            "single tumor cells or tiny clusters in lymph node",
            "very small isolated tumor cell cluster less than 0.2 mm",
            "isolated tumor cells not counted as lymph node metastasis",
        ),
        (
            "lymph node micrometastasis",
            "small metastatic breast carcinoma deposit in sentinel lymph node",
            "micrometastatic tumor deposit between 0.2 mm and 2.0 mm",
            "microscopic breast cancer metastasis in lymph node",
        ),
        (
            "lymph node macrometastasis",
            "large metastatic breast carcinoma deposit in sentinel lymph node",
            "macrometastatic tumor deposit greater than 2.0 mm",
            "overt breast cancer metastasis in lymph node",
        ),
    ),
    "brca": (
        (
            "invasive ductal carcinoma",
            "breast invasive ductal carcinoma",
            "invasive ductal carcinoma of the breast",
            "invasive carcinoma of the breast, ductal pattern",
            "idc",
        ),
        (
            "invasive lobular carcinoma",
            "breast invasive lobular carcinoma",
            "invasive lobular carcinoma of the breast",
            "invasive carcinoma of the breast, lobular pattern",
            "ilc",
        ),
    ),
    "rcc": (
        (
            "clear cell renal cell carcinoma",
            "renal cell carcinoma, clear cell type",
            "renal cell carcinoma of the clear cell type",
            "clear cell rcc",
        ),
        (
            "papillary renal cell carcinoma",
            "renal cell carcinoma, papillary type",
            "renal cell carcinoma of the papillary type",
            "papillary rcc",
        ),
        (
            "chromophobe renal cell carcinoma",
            "renal cell carcinoma, chromophobe type",
            "renal cell carcinoma of the chromophobe type",
            "chromophobe rcc",
        ),
    ),
    "nsclc": (
        ("adenocarcinoma", "lung adenocarcinoma", "adenocarcinoma of the lung", "luad"),
        (
            "squamous cell carcinoma",
            "lung squamous cell carcinoma",
            "squamous cell carcinoma of the lung",
            "lusc",
        ),
    ),
    "esca": (
        ("adenocarcinoma", "esophageal adenocarcinoma", "adenocarcinoma of the esophagus", "esad"),
        (
            "squamous cell carcinoma",
            "esophageal squamous cell carcinoma",
            "squamous cell carcinoma of the esophagus",
            "essc",
        ),
    ),
    "tgct": (
        ("seminoma", "testicular seminoma", "seminoma of the testis"),
        (
            "mixed germ cell tumor",
            "testicular mixed germ cell tumor",
            "mixed germ cell tumor of the testis",
        ),
    ),
    "cesc": (
        ("adenocarcinoma", "cervical adenocarcinoma", "adenocarcinoma of the cervix uteri"),
        (
            "squamous cell carcinoma",
            "cervical squamous cell carcinoma",
            "squamous cell carcinoma of the cervix uteri",
        ),
    ),
    "bracs": (
        (
            "benign breast lesion in BRACS histology",
            "normal or benign breast tissue",
            "pathological benign breast lesion",
            "usual ductal hyperplasia or benign breast change",
        ),
        (
            "atypical breast lesion in BRACS histology",
            "flat epithelial atypia or atypical ductal hyperplasia",
            "breast epithelial atypia without invasive carcinoma",
            "premalignant atypical breast lesion",
        ),
        (
            "malignant breast lesion in BRACS histology",
            "ductal carcinoma in situ or invasive breast carcinoma",
            "breast carcinoma lesion",
            "malignant epithelial breast tumor",
        ),
    ),
    "herohe": (
        (
            "HER2-negative",
            "HER2 negative invasive breast cancer",
            "breast carcinoma with absent HER2 overexpression",
            "HER2 non-amplified breast carcinoma",
            "invasive breast tumor with negative HER2 receptor status",
        ),
        (
            "HER2-positive",
            "HER2 positive invasive breast cancer",
            "breast carcinoma with HER2 overexpression",
            "HER2 amplified breast carcinoma",
            "invasive breast tumor with positive HER2 receptor status",
        ),
    ),
    "ubc_ocean": (
        (
            "ovarian high grade serous carcinoma",
            "high grade serous ovarian carcinoma",
            "HGSC ovarian carcinoma",
            "high grade serous carcinoma of the ovary",
        ),
        (
            "ovarian endometrioid carcinoma",
            "endometrioid ovarian carcinoma",
            "endometrioid carcinoma of the ovary",
        ),
        (
            "ovarian clear cell carcinoma",
            "clear cell ovarian carcinoma",
            "clear cell carcinoma of the ovary",
        ),
        (
            "ovarian low grade serous carcinoma",
            "low grade serous ovarian carcinoma",
            "LGSC ovarian carcinoma",
            "low grade serous carcinoma of the ovary",
        ),
        (
            "ovarian mucinous carcinoma",
            "mucinous ovarian carcinoma",
            "mucinous carcinoma of the ovary",
        ),
    ),
}

DEFAULT_TASK_ORDER: Tuple[str, ...] = tuple(PROMPT_REGISTRY)


def resolve_class_prompts(
    task_order: Sequence[str],
    task_num_classes: Sequence[int] | None = None,
) -> List[List[str]]:
    """Return flattened class synonym lists in the requested task order."""
    order = [str(task_name) for task_name in task_order]
    if len(set(order)) != len(order):
        raise ValueError(f"task_order contains duplicates: {order}")
    if task_num_classes is not None and len(task_num_classes) != len(order):
        raise ValueError("task_order and task_num_classes must have equal lengths")

    flattened: List[List[str]] = []
    for task_id, task_name in enumerate(order):
        if task_name not in PROMPT_REGISTRY:
            raise ValueError(f"No QPMIL-VL prompts registered for task {task_name!r}")
        task_prompts = PROMPT_REGISTRY[task_name]
        if task_num_classes is not None:
            expected = int(task_num_classes[task_id])
            if len(task_prompts) != expected:
                raise ValueError(
                    f"Task {task_name!r} has {len(task_prompts)} prompt classes; "
                    f"dataset declares {expected}"
                )
        if any(not synonyms or any(not text.strip() for text in synonyms) for synonyms in task_prompts):
            raise ValueError(f"Task {task_name!r} contains an empty class prompt")
        flattened.extend([list(synonyms) for synonyms in task_prompts])
    return flattened


def prompt_schema_hash(
    task_order: Sequence[str],
    task_num_classes: Sequence[int] | None = None,
) -> str:
    """Hash the exact ordered prompt schema stored outside checkpoints."""
    payload = {
        "task_order": [str(value) for value in task_order],
        "class_prompts": resolve_class_prompts(task_order, task_num_classes),
        "templates": list(TEMPLATES),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_default_registry() -> None:
    """Fail if the checked-in stream registry ceases to cover exactly 27 classes."""
    prompts = resolve_class_prompts(DEFAULT_TASK_ORDER)
    if len(DEFAULT_TASK_ORDER) != 10 or len(prompts) != 27:
        raise ValueError(
            "QPMIL-VL prompt registry must cover 10 tasks and 27 classes; "
            f"got {len(DEFAULT_TASK_ORDER)} tasks and {len(prompts)} classes"
        )


validate_default_registry()
