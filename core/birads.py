"""
BI-RADS-style categories and referral pathways.

BI-RADS (Breast Imaging Reporting and Data System, American College of Radiology, 5th edition)
assigns every breast-imaging finding to one assessment category, each with an expected likelihood
of malignancy and a standard management recommendation:

    0  Incomplete: needs additional imaging evaluation and/or prior images for comparison
    1  Negative                                    essentially 0 %
    2  Benign                                      essentially 0 %
    3  Probably benign                             > 0 % and <= 2 %
    4A Low suspicion for malignancy                > 2 % and <= 10 %
    4B Moderate suspicion for malignancy           > 10 % and <= 50 %
    4C High suspicion for malignancy               > 50 % and < 95 %
    5  Highly suggestive of malignancy             >= 95 %
    6  Known biopsy-proven malignancy

IMPORTANT. A BI-RADS category is assigned by a radiologist who reads the whole examination.
Here the category is *suggested* by mapping the AI model's malignancy score onto the BI-RADS
likelihood bands. The mammogram model was trained on CBIS-DDSM, where every image contains a
lesion that was considered suspicious enough to work up, so its score is not calibrated to a
screening population. Treat the output as decision support for a clinician, never as a diagnosis.
"""
from __future__ import annotations

CATEGORIES = {
    "0":  {"name": "Incomplete", "likelihood": "n/a", "level": 0},
    "1":  {"name": "Negative", "likelihood": "≈ 0%", "level": 1},
    "2":  {"name": "Benign", "likelihood": "≈ 0%", "level": 1},
    "3":  {"name": "Probably benign", "likelihood": "≤ 2%", "level": 2},
    "4A": {"name": "Low suspicion for malignancy", "likelihood": "2–10%", "level": 3},
    "4B": {"name": "Moderate suspicion for malignancy", "likelihood": "10–50%", "level": 4},
    "4C": {"name": "High suspicion for malignancy", "likelihood": "50–95%", "level": 5},
    "5":  {"name": "Highly suggestive of malignancy", "likelihood": "≥ 95%", "level": 6},
    "6":  {"name": "Known biopsy-proven malignancy", "likelihood": "100%", "level": 6},
}

# Urgency levels used across the app (shown as a coloured badge)
URGENCY = {
    "routine":   {"label": "Routine", "timeframe": "Next scheduled screening"},
    "followup":  {"label": "Follow-up", "timeframe": "Repeat imaging in 6 months"},
    "recall":    {"label": "Recall", "timeframe": "Additional imaging within 2–4 weeks"},
    "soon":      {"label": "Referral", "timeframe": "Specialist review & biopsy within 2–4 weeks"},
    "urgent":    {"label": "Urgent referral", "timeframe": "Specialist review within 2 weeks"},
    "immediate": {"label": "Urgent: oncology", "timeframe": "Multidisciplinary team within 1–2 weeks"},
}

MANAGEMENT = {
    "0": {
        "urgency": "recall",
        "summary": "Additional imaging needed before a final assessment can be made.",
        "actions": [
            "Recall for diagnostic mammographic views (spot compression / magnification) and/or targeted breast ultrasound.",
            "Obtain prior mammograms for comparison if available.",
            "Re-assign a final BI-RADS category (1–5) after the work-up.",
        ],
        "refer_to": "Breast imaging / radiology department",
    },
    "1": {
        "urgency": "routine",
        "summary": "No abnormality seen. Continue routine screening.",
        "actions": ["Return to routine screening according to age and risk (typically every 1–2 years from age 40–50)."],
        "refer_to": "No referral needed",
    },
    "2": {
        "urgency": "routine",
        "summary": "Benign finding. Continue routine screening.",
        "actions": ["Document the benign finding.", "Return to routine screening according to age and risk."],
        "refer_to": "No referral needed",
    },
    "3": {
        "urgency": "followup",
        "summary": "Probably benign. Short-interval imaging follow-up rather than biopsy.",
        "actions": [
            "Diagnostic mammogram (± ultrasound) of the affected breast in 6 months.",
            "If stable, further follow-up at 12 and 24 months (2 years in total).",
            "Biopsy instead if the finding grows, the patient is high-risk, or follow-up cannot be guaranteed.",
        ],
        "refer_to": "Breast imaging follow-up clinic",
    },
    "4A": {
        "urgency": "soon",
        "summary": "Low suspicion. Tissue diagnosis is recommended.",
        "actions": [
            "Image-guided core needle biopsy (ultrasound- or stereotactic-guided).",
            "Refer to a breast clinic / breast surgeon for assessment.",
            "Check radiology–pathology concordance after biopsy; a benign concordant result usually returns to 6-month follow-up.",
        ],
        "refer_to": "Breast clinic (surgeon + radiologist)",
    },
    "4B": {
        "urgency": "urgent",
        "summary": "Moderate suspicion. Biopsy and specialist referral are recommended.",
        "actions": [
            "Image-guided core needle biopsy.",
            "Urgent referral to a specialist breast clinic for triple assessment (clinical exam, imaging, pathology).",
            "Ensure radiology–pathology concordance; discordant benign results need repeat biopsy or excision.",
        ],
        "refer_to": "Specialist breast clinic (urgent)",
    },
    "4C": {
        "urgency": "urgent",
        "summary": "High suspicion. Prompt biopsy and urgent specialist referral.",
        "actions": [
            "Image-guided core needle biopsy as soon as possible.",
            "Urgent referral to a specialist breast clinic (target: seen within 2 weeks).",
            "Prepare for multidisciplinary team (MDT) discussion if malignancy is confirmed.",
        ],
        "refer_to": "Specialist breast clinic (urgent, ≤ 2 weeks)",
    },
    "5": {
        "urgency": "immediate",
        "summary": "Highly suggestive of malignancy. Biopsy and urgent oncology pathway.",
        "actions": [
            "Image-guided core needle biopsy to confirm the diagnosis and receptor status (ER/PR/HER2).",
            "Urgent referral to the breast cancer multidisciplinary team (surgery, oncology, radiology, pathology).",
            "Consider staging work-up and axillary ultrasound per specialist advice.",
        ],
        "refer_to": "Breast cancer MDT / surgical oncology (urgent)",
    },
    "6": {
        "urgency": "immediate",
        "summary": "Known malignancy. Treatment planning by the oncology team.",
        "actions": [
            "Multidisciplinary team review for staging and treatment planning.",
            "Receptor status (ER/PR/HER2) and grade guide systemic therapy and surgery.",
        ],
        "refer_to": "Breast cancer MDT / oncology",
    },
}

def score_bands(thr: float) -> list:
    """Score cut-offs for each category, anchored on the model's validated decision threshold `thr`
    (chosen on the validation set by Youden's J). The raw model score is NOT a calibrated likelihood
    (CBIS-DDSM only contains suspicious, worked-up findings), so the ACR percentage bands cannot be
    applied to it directly. Instead:
        below the threshold  -> 2 (lower half) or 3 (upper half)
        above the threshold  -> 4A / 4B / 4C / 5 by how far the score is towards 1.0
    """
    t = min(max(thr, 0.05), 0.95)
    r = 1.0 - t
    return [(0.5 * t, "2"), (t, "3"), (t + 0.25 * r, "4A"), (t + 0.55 * r, "4B"), (t + 0.85 * r, "4C"), (1.01, "5")]


def category_from_score(p: float, thr: float) -> str:
    for upper, cat in score_bands(thr):
        if p < upper:
            return cat
    return "5"


def _pack(cat: str, basis: str, extra_note: str | None = None, thr: float | None = None) -> dict:
    m = MANAGEMENT[cat]
    out = {
        "system": "BI-RADS", "category": cat, **CATEGORIES[cat],
        "urgency": m["urgency"], **{f"urgency_{k}": v for k, v in URGENCY[m["urgency"]].items()},
        "summary": m["summary"], "actions": m["actions"], "refer_to": m["refer_to"], "basis": basis,
        "bands": [{"category": c, **CATEGORIES[c]} for c in ("0", "1", "2", "3", "4A", "4B", "4C", "5", "6")],
    }
    if thr is not None:
        lo = 0.0
        cut = []
        for upper, c in score_bands(thr):
            cut.append({"category": c, "from": round(lo, 3), "to": round(min(upper, 1.0), 3)})
            lo = upper
        out["score_bands"] = cut
        out["threshold"] = thr
    if extra_note:
        out["note"] = extra_note
    return out


def assess_mammogram(p: float, thr: float = 0.5, image_ok: bool = True, quality_issue: str | None = None) -> dict:
    """AI-suggested BI-RADS category for a mammogram finding."""
    if not image_ok:
        return _pack("0", "Image could not be assessed reliably by the model.",
                     quality_issue or "The breast could not be located or the image quality is too low.")
    cat = category_from_score(p, thr)
    return _pack(cat, f"Model score {p:.1%} (decision threshold {thr:.1%}) placed in the score band for BI-RADS {cat}.",
                 "Suggested category only. The final BI-RADS assessment must be made by a radiologist reading the "
                 "full examination (both views, both breasts, prior studies). The model was trained only on images "
                 "that contain a finding, so it never suggests BI-RADS 1 (negative).", thr)


def assess_generic() -> dict:
    return _pack("0", "No trained model for this image type, so the image analysis cannot assign a category.",
                 "MRI and ultrasound also use BI-RADS, but this app has no model trained on them. "
                 "A radiologist should review the study.")


def assess_histology(detected: bool, score: float, fraction: float) -> dict:
    """Pathology is not graded with BI-RADS: give the pathology pathway instead."""
    if detected:
        return {
            "system": "Pathology", "category": "IDC+", "name": "Invasive ductal carcinoma pattern detected",
            "likelihood": f"{fraction:.0%} of tissue patches flagged", "level": 6,
            "urgency": "immediate", **{f"urgency_{k}": v for k, v in URGENCY["immediate"].items()},
            "summary": "Tissue pattern consistent with invasive ductal carcinoma. Needs pathologist confirmation.",
            "actions": [
                "Pathologist review of the slide to confirm the diagnosis and grade.",
                "If confirmed (equivalent to BI-RADS 6, known malignancy): receptor testing (ER/PR/HER2, Ki-67).",
                "Urgent referral to the breast cancer multidisciplinary team for staging and treatment planning.",
                "Check concordance with the imaging findings.",
            ],
            "refer_to": "Pathology confirmation → breast cancer MDT / oncology",
            "basis": f"Slide score {score:.1%} (mean of the most suspicious patches).",
            "note": "Histopathology is not graded with BI-RADS; once a malignancy is biopsy-proven the imaging category becomes BI-RADS 6.",
        }
    return {
        "system": "Pathology", "category": "IDC−", "name": "No invasive ductal carcinoma pattern detected",
        "likelihood": f"{fraction:.0%} of tissue patches flagged", "level": 1,
        "urgency": "routine", **{f"urgency_{k}": v for k, v in URGENCY["routine"].items()},
        "summary": "No IDC pattern found by the model. Confirm with the pathologist and the imaging.",
        "actions": [
            "Pathologist review to confirm a benign result.",
            "Radiology–pathology concordance: if the imaging was BI-RADS 4 or 5, a benign result may be discordant and need repeat biopsy or excision.",
            "If concordant and benign: return to the follow-up advised by the radiologist.",
        ],
        "refer_to": "Pathologist review; follow imaging recommendation",
        "basis": f"Slide score {score:.1%}.",
        "note": "The model only detects invasive ductal carcinoma; other lesion types (e.g. lobular carcinoma, DCIS) are outside its training.",
    }


def assess_histology_inadequate() -> dict:
    return {
        "system": "Pathology", "category": "N/A", "name": "Not assessable: no tissue found", "likelihood": "no tissue patches",
        "level": 0, "urgency": "recall", "urgency_label": "Resubmit", "urgency_timeframe": "Upload a stained tissue image",
        "summary": "The image contains no recognisable H&E tissue, so it could not be analysed.",
        "actions": ["Check that the image is a stained histology slide (not a mammogram or other scan).",
                    "If it is, rescan or upload a region that contains tissue."],
        "refer_to": "Repeat upload / pathology lab", "basis": "No tissue patches passed the tissue filter.",
        "note": "Choose the matching scan type at the top of the Scan Lab.",
    }


def assess_cytology(p: float) -> dict:
    """Digital Twin (fine-needle aspirate measurements): cytology-style pathway."""
    if p >= 0.85:
        cat, name, urg = "C5", "Malignant features", "immediate"
        summary = "Nuclear features strongly suggest malignancy."
        actions = ["Core needle biopsy for histological confirmation and receptor status.",
                   "Urgent referral to the breast cancer multidisciplinary team."]
        refer = "Breast cancer MDT (urgent)"
    elif p >= 0.5:
        cat, name, urg = "C4", "Suspicious of malignancy", "urgent"
        summary = "Features are suspicious but not conclusive."
        actions = ["Core needle biopsy (FNA alone is not sufficient).",
                   "Urgent referral to a specialist breast clinic for triple assessment."]
        refer = "Specialist breast clinic (urgent)"
    elif p >= 0.15:
        cat, name, urg = "C3", "Atypia, probably benign", "soon"
        summary = "Indeterminate features. A benign result is likely but not certain."
        actions = ["Correlate with imaging (mammogram / ultrasound BI-RADS).",
                   "Core needle biopsy if imaging or clinical findings are suspicious."]
        refer = "Breast clinic"
    else:
        cat, name, urg = "C2", "Benign", "routine"
        summary = "Nuclear features are consistent with a benign lesion."
        actions = ["Confirm concordance with clinical examination and imaging (triple assessment).",
                   "Routine follow-up if all three are benign."]
        refer = "No urgent referral; routine follow-up"
    return {
        "system": "Cytology", "category": cat, "name": name, "likelihood": f"{p:.0%} model probability",
        "urgency": urg, **{f"urgency_{k}": v for k, v in URGENCY[urg].items()},
        "summary": summary, "actions": actions, "refer_to": refer,
        "basis": "Digital Twin SVM probability from the 30 fine-needle-aspirate measurements.",
        "note": "Categories follow the UK C1–C5 cytology scheme. They are assigned by a cytopathologist; this is a model-based suggestion.",
    }
