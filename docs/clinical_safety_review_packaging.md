# Clinical Safety Review Packaging & Risk Governance

## 1. Overview & Scope Limitation

Per Process Integrity rules (P.3), this document packages the clinical safety architecture of the HMIS for review by qualified medical officers, clinical pharmacists, and healthcare governance committees. **This document does not constitute clinical safety certification.**

---

## 2. Clinical Decision Support System (CDSS) Scope & Limitations

### 2.1 Drug-Drug Interaction (DDI) Engine
* **Active Scope**:
  - Live querying against **DrugCentral PostgreSQL Database** (7,621 DDI rules) based on drug structure and pharmacological class matching.
  - Fail-safe fallback matrix (`KNOWN_INTERACTIONS` in `departments/medicine/cdss.py`) covering high-consequence pairs (e.g. Warfarin + Aspirin, Sildenafil + Nitroglycerin, Lisinopril + Spironolactone).
* **Limitations**:
  - Does not substitute for complete clinical pharmacology evaluation.
  - Complex multi-drug regimens (> 5 concurrent drugs) require manual pharmacist review.

### 2.2 Patient Allergy Screening
* **Active Scope**:
  - Matches prescribed drug names against recorded patient allergy history (e.g. Penicillin, Sulfa, NSAID classes).
* **Limitations**:
  - Relies on structured allergy notes; free-text unstructured notes may omit subtle cross-sensitivities.

### 2.3 Renal / Hepatic Dosing Adjustments
* **Active Scope**:
  - Evaluates patient eGFR / CrCl against threshold guidelines for narrow therapeutic index drugs (`Metformin`, `Gentamicin`, `Vancomycin`, `Enoxaparin`).
* **Limitations**:
  - Requires up-to-date serum creatinine values in patient record.

---

## 3. Recommended Audit Questions for Clinical Review Board

1. **Alert Fatigue Thresholds**:
   - Are the severity levels (HIGH vs MODERATE) appropriate to prevent clinician alert fatigue while ensuring critical warnings are acted upon?
2. **Local Formulary Alignment**:
   - Does the local DrugCentral mapping reflect the specific essential medicine list (KEMSA / Kenya National Formulary) used by facility clinicians?
3. **Override & Audit Policy**:
   - When a clinician overrides a CDSS warning, is the required text justification logged and reviewed periodically by the Pharmacy & Therapeutics Committee?
