# Prompt 2 of 9 — Controlled Drug Register (Phase 3)

Priority: Legal blocker. Schedule II narcotics (morphine, pethidine, fentanyl) and Schedule IV
psychotropics (diazepam, lorazepam, midazolam) cannot be legally dispensed without a
dual-signature register under Kenya Pharmacy and Poisons Board (PPB) regulation.

## Stop condition — read this before doing anything else

Three decisions in `DECISIONS_PENDING.md` item 5 are **not yet resolved**:
1. Which roles qualify as the mandatory second signatory (two licensed pharmacists, or
   pharmacist + ward nurse-in-charge)?
2. Stock reconciliation schedule: per shift, daily, or weekly?
3. Separate ledgers for Schedule II vs Schedule IV, or a unified controlled log?

**Do not write dispensing logic that enforces an assumed answer to any of these.** This is a
real legal-compliance workflow, not a placeholder to fill in and adjust later — a wrong
default here is a compliance incident, not a bug.

## Part A — Decision drafting (do this first)

Produce a short proposal document (add it to the repo as
`docs/decisions/phase3_controlled_drug_register_proposal.md`) that:
- Lays out the trade-offs for each of the three open decisions above, in plain terms a
  hospital administrator or pharmacy lead can act on.
- Notes that the current PPB Controlled Drug Register format must be obtained directly from
  the PPB compliance officer — do not assume a format from historic documents or general
  knowledge of pharmacy regulation.
- Ends with a clear "needs sign-off from: [Clinical Lead / Pharmacy Lead]" line.

Stop here and wait for the human to record a decision in `DECISIONS_PENDING.md` item 5 before
proceeding to Part B. If asked to proceed with Part B in the same session, only do so if the
three decisions above are explicitly resolved in the prompt or in `DECISIONS_PENDING.md`.

## Part B — Implementation (only after decisions are recorded)

Branch: `feat/phase3-controlled-drug-register`. Never commit to `main` directly, never
force-push.

| Item | Acceptance criteria | Notes |
|------|---------------------|-------|
| P3-01 | Add `ControlledDrug` boolean + `schedule_class` (II/IV/V) column to `Drug` model. Seed classification for all current formulary drugs. Alembic migration required. | |
| P3-02 | Build `ControlledDrugDispense` model: `patient_id`, `drug_id`, `dose_mg`, `dispense_datetime`, `primary_pharmacist_id`, `second_signatory_id`, `second_signatory_role`, `balance_after`, `witness_signature_hash`. | |
| P3-03 | Enforce dual-signature workflow in dispensing UI: after primary pharmacist confirms, system locks the dispense and requires a second named signatory. Neither person can sign for both roles. | Implement per the resolved decision on who qualifies as second signatory. |
| P3-04 | Build running balance ledger: each dispense and receipt updates a `ControlledDrugBalance` row. Any negative balance triggers a CRITICAL alert to the pharmacy supervisor. | |
| P3-05 | Build shift reconciliation screen: pharmacist enters physical count, system computes variance vs. ledger balance, flags discrepancies > 0 for immediate investigation. | Cadence per the resolved reconciliation-schedule decision. |
| P3-06 | Build PPB-format printed register: exportable PDF matching the official PPB Controlled Drugs Register format (obtained in Part A, not assumed). | |
| P3-07 | Tests: dispense without second signatory returns 403; negative balance alert fires; shift reconciliation variance > 0 blocks close. | |
| P3-08 | Flag for human sign-off: Clinical Lead must review the workflow against PPB Pharmacy Act Cap 244 requirements before any live dispense. Do not mark this item complete yourself — leave it open in the PR description. | |

## Known risk to handle explicitly

If no second signatory is available on a night shift, the system must **not** allow the
dispense to silently proceed. The on-call clinical officer must be documented as the second
signatory, and the workflow must prompt for this rather than failing open or failing silently.

## Done when

- Full test suite passes; ruff clean.
- P3-08's human review is explicitly called out as pending in the PR, not marked done.
- Branch pushed (not force-pushed), ready for PR.
