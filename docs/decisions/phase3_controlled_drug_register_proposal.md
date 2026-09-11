# Phase 3 Proposal: Controlled Drug Register (PPB Compliance)

**Status:** Awaiting Human Sign-Off
**Priority:** Legal Blocker (Kenya Pharmacy and Poisons Board Regulation)
**Scope:** Schedule II narcotics (morphine, pethidine, fentanyl) and Schedule IV psychotropics (diazepam, lorazepam, midazolam).

## Context
Under Kenya PPB regulation, controlled drugs cannot be legally dispensed without a dual-signature register. The current system lacks the workflow enforcement, balance tracking, and shift reconciliation required for compliance. Implementing this incorrectly (e.g., failing open on night shifts) creates immediate legal exposure.

Before any code is written for Phase 3 (Work items P3-01 through P3-08), the following three operational decisions must be resolved by hospital leadership.

---

## Decision 1: Mandatory Second Signatory Roles
**Question:** Who qualifies as the mandatory second signatory when a controlled drug is dispensed?

*   **Option A: Two Licensed Pharmacists Only**
    *   *Pros:* Highest level of control; strictly aligns with the most conservative reading of PPB guidelines.
    *   *Cons:* Operationally impossible on night shifts or weekends when only one pharmacist is on duty. Would block emergency dispensing (e.g., pethidine for labor) unless a second pharmacist is called in.
*   **Option B: Pharmacist + Ward Nurse-in-Charge (or Clinical Officer)**
    *   *Pros:* Reflects clinical reality. A nurse-in-charge is physically present 24/7 and can verify the drug was administered to the correct patient.
    *   *Cons:* Slightly lower chain-of-custody strictness than two pharmacists; requires defining exactly which nursing grades qualify.
*   **Option C: Pharmacist + On-Call Clinical Officer (for night shifts only)**
    *   *Pros:* Balances security with 24/7 availability.
    *   *Cons:* Complex workflow logic (role depends on time of day).

**Recommendation:** Option B is standard in most Tier 1 Kenyan hospitals for practical compliance, provided the nurse's PIN is recorded.

---

## Decision 2: Stock Reconciliation Schedule
**Question:** How often must the physical count be reconciled against the system ledger?

*   **Option A: Per Shift (Every 8 Hours)**
    *   *Pros:* Rapid detection of diversion or errors. Matches physical shift handover protocols.
    *   *Cons:* High administrative burden on nursing/pharmacy staff (3x daily counts).
*   **Option B: Daily (Once per 24 Hours)**
    *   *Pros:* Standard practice for most wards. Manageable workload.
    *   *Cons:* Diversion or errors may go undetected for up to 24 hours.
*   **Option C: Weekly**
    *   *Pros:* Low burden.
    *   *Cons:* Unacceptable for Schedule II narcotics under PPB guidelines. High risk of regulatory penalty.

**Recommendation:** Option A (Per Shift) for Schedule II (narcotics); Option B (Daily) for Schedule IV (psychotropics).

---

## Decision 3: Ledger Structure (Unified vs. Separate)
**Question:** Should the system maintain separate digital ledgers for Schedule II and Schedule IV, or a unified controlled log?

*   **Option A: Separate Ledgers (Two distinct tables/views)**
    *   *Pros:* Mirrors the physical PPB register books (which are physically separate). Easier for auditors to verify "book vs. system".
    *   *Cons:* More complex database schema and UI navigation for pharmacists.
*   **Option B: Unified Controlled Log (Single table with `schedule_class` filter)**
    *   *Pros:* Better software UX; easier to search across all controlled drugs.
    *   *Cons:* Requires strict filtering to print/export the separate physical registers.

**Recommendation:** Option A (Separate Ledgers) to ensure 1:1 mapping with physical PPB audit books.

---

## ⚠️ Critical Dependency: PPB Register Format
**Note:** The exact column layout, page numbering format, and required fields for the printed PPB Controlled Drug Register (Work Item P3-06) **must be obtained directly from the PPB compliance officer**. 

*Do not assume a format based on historic documents, general pharmacy knowledge, or other hospitals' implementations.* The format is subject to change and must be verified against the current *Pharmacy and Poisons Act (Cap 244)* guidelines before the PDF export logic is built.

---

## Sign-Off Required
**This proposal must be reviewed and the three decisions above must be recorded in `DECISIONS_PENDING.md` (Item 5) before Phase 3 implementation begins.**

**Needs sign-off from:**
[ ] Clinical Lead / Medical Superintendent
[ ] Pharmacy Lead / Chief Pharmacist

**Date:** _______________
**Signatures:** _______________
