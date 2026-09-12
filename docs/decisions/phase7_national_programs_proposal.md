# Phase 7 National Disease Programs Proposal
## HIV/ART, TB, and Malaria Modules

### Reference: MCH/ANC Module Structure
Before proposing the disease-specific modules, we examine the existing MCH/ANC module (`departments/mch/`) as the structural reference for depth and complexity:

**MCH/ANC Data Models:**
- `AncVisit`: Patient visits with gestation tracking, risk factors, appointment scheduling
- `ImmunizationRecord`: Vaccination records with batch tracking and administration details

**MCH/ANC Features:**
- Visit-based workflow tracking
- Integration with encounters (FK to encounters table)
- Timezone-aware datetime tracking
- UUID-based primary keys
- Indexed foreign keys for performance
- Default values for timestamps

---

## HIV/ART Module Proposal

### Proposed Data Model Structure
Using MCH/ANC as reference, the HIV/ART module should include:

1. **ARTEnrollment** (similar to AncVisit)
   - patient_id (String, indexed, FK to patients)
   - enrollment_date (DateTime, timezone-aware)
   - art_number (String, unique identifier)
   - baseline_cd4 (Integer)
   - baseline_who_stage (Integer, 1-4)
   - art_start_date (DateTime)
   - current_regimen (String, FK to ARTRegimen)
   - facility_enrolled_at (String)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

2. **ARTRegimen** (reference table, not a transactional model)
   - regimen_code (String, primary key)
   - regimen_name (String)
   - line_of_therapy (Integer, 1, 2, 3+)
   - arv_drugs (String, comma-separated list or JSON)
   - is_preferred (Boolean)
   - is_alternative (Boolean)
   - restriction_notes (Text, for contraindications)
   - effective_from (Date)
   - effective_to (Date, nullable)

3. **AdherenceVisit** (similar to AncVisit but for ART)
   - patient_id (String, indexed)
   - visit_date (DateTime, timezone-aware)
   - pills_dispensed (Integer)
   - pills_returned (Integer)
   - days_since_last_visit (Integer)
   - adherence_percentage (Float, calculated)
   - adherence_category (String: good/fair/poor)
   - viral_load_ordered (Boolean)
   - cd4_ordered (Boolean)
   - next_visit_date (Date)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

4. **ViralLoad** (lab result tracking)
   - patient_id (String, indexed)
   - test_date (DateTime, timezone-aware)
   - viral_load_copies (Integer)
   - test_type (String: routine, diagnostic, confirmation)
   - result_date (Date)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

5. **CD4Count** (immunological tracking)
   - patient_id (String, indexed)
   - test_date (DateTime, timezone-aware)
   - cd4_count (Integer, cells/µL)
   - cd4_percent (Float, optional)
   - test_date (Date)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

6. **WHOStage** (clinical staging)
   - patient_id (String, indexed)
   - assessment_date (DateTime, timezone-aware)
   - who_stage (Integer, 1-4)
   - defining_conditions (Text)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

### Open Clinical Questions for HIV Clinician Review
1. What specific ART regimen definitions should be used from the MOH Kenya ART formulary?
2. How should treatment interruptions be tracked and classified?
3. What are the criteria for switching from first-line to second-line regimens?
4. How should pediatric ART dosing and monitoring differ from adult protocols?
5. What is the preferred frequency for viral load and CD4 monitoring per MOH guidelines?
6. How should pregnancy status and safety of ARVs in pregnancy be tracked?
7. What definitions of treatment failure should be used (virological, immunological, clinical)?

### Proposed Clinical Reviewer
To be determined by facility management - should be an HIV clinician familiar with:
- MOH Kenya ART guidelines
- Adult and pediatric ART management
- Treatment failure criteria
- PMTCT (Prevention of Mother-to-Child Transmission) protocols

---

## TB/DOTS Module Proposal

### Proposed Data Model Structure
Using MCH/ANC as reference, the TB/DOTS module should include:

1. **TBEnrollment** (similar to AncVisit)
   - patient_id (String, indexed, FK to patients)
   - enrollment_date (DateTime, timezone-aware)
   - tb_case_number (String, unique identifier)
   - tb_classification (String: pulmonary, extrapulmonary)
   - bacteriological_status (String: confirmed, clinically diagnosed)
   - hiv_status (String: positive, negative, unknown)
   - registration_date (DateTime)
   - treatment_start_date (DateTime)
   - facility_registered_at (String)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

2. **DOTSPhase** (tracking intensive/continuation phases)
   - patient_id (String, indexed)
   - phase_start_date (DateTime, timezone-aware)
   - phase_end_date (DateTime, timezone-aware, nullable)
   - phase_type (String: intensive, continuation)
   - regimen_used (String, FK to TBRegimen reference)
   - doses_due_this_phase (Integer)
   - doses_taken_this_phase (Integer)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

3. **TBRegimen** (reference table)
   - regimen_code (String, primary key)
   - regimen_name (String)
   - phase_applicable (String: intensive, continuation, both)
   - drugs_in_regimen (String, JSON or comma-separated)
   - duration_months (Integer)
   - is_first_line (Boolean)
   - is_retreatment (Boolean)
   - restriction_notes (Text)
   - effective_from (Date)
   - effective_to (Date, nullable)

4. **SputumResult** (diagnostic tracking)
   - patient_id (String, indexed)
   - specimen_date (DateTime, timezone-aware)
   - specimen_type (String: sputum, gastric aspirate, etc.)
   - test_method (String: microscopy, Xpert MTB/RIF, culture)
   - result (String: negative, scanty, 1+, 2+, 3+)
   - result_date (Date)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

5. **XpertResult** (molecular diagnostic)
   - patient_id (String, indexed)
   - test_date (DateTime, timezone-aware)
   - result (String: MTB detected/not detected, rifampicin resistance detected/not detected)
   - ct_value (Float, optional)
   - error_code (String, nullable)
   - test_date (Date)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

6. **DrugSensitivity** (for MDR-TB tracking)
   - patient_id (String, indexed)
   - test_date (DateTime, timezone-aware)
   - tested_drugs (String: INH, RIF, EMB, STM, etc.)
   - resistance_pattern (String: susceptible, mono-resistance, poly-resistance, MDR, XDR)
   - method_used (String: phenotypic, genotypic)
   - result_date (Date)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

7. **TreatmentOutcome** (final outcome tracking)
   - patient_id (String, indexed)
   - outcome_date (DateTime, timezone-aware)
   - outcome_type (String: cured, treatment completed, treatment failed, died, lost to follow-up, not evaluated)
   - outcome_date (Date)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

### Open Clinical Questions for TB Clinician Review
1. What is the current MOH Kenya TB regimen for new smear-positive cases?
2. How should pediatric TB dosing and management be handled?
3. What are the criteria for diagnosing extrapulmonary TB?
4. How should TB-HIV co-infection be managed (timing of ART initiation)?
5. What is the protocol for contact tracing and preventive therapy (TPT)?
6. How should treatment adherence be measured and what thresholds trigger intervention?
7. What definitions of treatment failure, relapse, and recurrence should be used?
8. How should drug toxicity and side effects be monitored and managed?

### Proposed Clinical Reviewer
To be determined by facility management - should be a TB clinician familiar with:
- MOH Kenya TB guidelines
- DOTS strategy implementation
- TB-HIV co-infection management
- Pediatric TB management
- Drug-resistant TB management

---

## Malaria Module Proposal

### Proposed Data Model Structure
Using MCH/ANC as reference, the Malaria module should include:

1. **MalariaTest** (diagnostic encounter)
   - patient_id (String, indexed, FK to patients)
   - test_date (DateTime, timezone-aware)
   - test_type (String: RDT, microscopy)
   - specimen_type (String: venous blood, capillary blood)
   - parasite_species (String: falciparum, vivax, malariae, ovale, mixed)
   - parasite_density (String: negative, +, ++, +++, ++++ or parasites/µL for microscopy)
   - test_result (String: positive, negative)
   - technician_name (String)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

2. **MalariaTreatment** (prescription tracking)
   - patient_id (String, indexed)
   - prescription_date (DateTime, timezone-aware)
   - prescribed_by (String, FK to users)
   - antimalarial_drug (String: AL, ASAQ, quinine, artesunate, etc.)
   - dosage_mg_per_kg (Float)
   - total_dose_mg (Integer)
   - duration_days (Integer)
   - treatment_start_date (Date)
   - treatment_end_date (Date)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

3. **MalariaOutcome** (follow-up tracking)
   - patient_id (String, indexed)
   - follow_up_date (DateTime, timezone-aware)
   - symptoms_resolved (Boolean)
   - parasite_clearance (String: cleared, persistent, recrudescent)
   - adverse_reactions (Text)
   - referral_made (Boolean)
   - referral_facility (String, nullable)
   - encounter_id (Integer, FK to encounters, nullable, indexed)

### Open Clinical Questions for Malaria Specialist Review
1. What is the current MOH Kenya malaria treatment policy for uncomplicated falciparum malaria?
2. How should severe malaria be managed and when should referral occur?
3. What is the policy for malaria in pregnancy (IPTp, treatment)?
4. How should mixed infections be treated?
5. What is the approach to malaria diagnosis in areas with low transmission?
6. How should artemisinin resistance be monitored and managed?
7. What is the recommended follow-up schedule after treatment completion?
8. How should asymptomatic parasitemia be handled?

### Proposed Clinical Reviewer
To be determined by facility management - should be a malaria specialist or infectious disease physician familiar with:
- MOH Kenya malaria treatment guidelines
- Diagnosis and treatment of severe malaria
- Malaria in pregnancy (MIP) interventions
- Drug resistance monitoring
- Vector control integration

---

## Scope Recommendation

**Recommendation: Facility-relevant subset for initial deployment**

**Reasoning:**
1. **Resource Optimization**: Developing full modules for all three diseases requires significant clinical reviewer time and development effort. Starting with facility-relevant subset allows for faster delivery of value.
2. **Epidemiological Relevance**: Disease burden varies significantly by geographic region within Kenya. Facilities in low-malaria zones may prioritize HIV/TB, while high-burden areas may need all three.
3. **Iterative Development**: Allows for learning and refinement from initial implementation before scaling to additional diseases.
4. **Clinical Workflow Focus**: Ensures development effort aligns with actual patient volume and clinical needs at the facility.
5. **Risk Mitigation**: Reduces risk of building unused functionality if disease prevalence assumptions are incorrect.

**Implementation Approach:**
- Phase 7A: Implement HIV/ART module first (typically highest chronic disease burden requiring ongoing management)
- Phase 7B: Implement TB module second (requires intensive monitoring but finite treatment duration)
- Phase 7C: Implement Malaria module third (often acute, seasonal, and may have lower chronic management burden)
- Each phase proceeds only after clinical reviewer sign-off for that specific program

---

## DHIS2 Integration Timing Recommendation

**Recommendation: Build standalone first, integrate with DHIS2 exporter in subsequent iteration**

**Reasoning:**
1. **Clinical Validation First**: Ensures the core clinical workflow and data models are correct and usable by clinicians before adding reporting complexity.
2. **Simplified Initial Development**: Reduces coupling between clinical workflow and reporting requirements, allowing faster iteration on clinical features.
3. **Data Quality Assurance**: Allows time to validate data quality and completeness at the source before exporting to national systems.
4. **Flexibility in Reporting Requirements**: DHIS2 indicators and data elements may evolve; standalone build allows adaptation to final requirements.
5. **Reduced Blockers**: Avoids dependency on DHIS2 exporter readiness or changes for clinical module development.

**Implementation Approach:**
- Develop each disease module with complete clinical workflow, data models, and local reporting/export capabilities
- Design data models with future DHIS2 mapping in mind (using standard codes where possible)
- After clinical validation and facility use, extend the existing DHIS2 exporter (`departments/api/dhis2_exporter.py`) to include:
  - HIV/ART: MOH 731 ART cohort report (monthly)
  - TB: MOH TB register (quarterly)
  - Malaria: MOH 705A/B malaria cases (monthly)
- Use the same patterns already established in the exporter for other modules

---

## Next Steps
1. **Clinical Reviewer Assignment**: Facility management to assign named clinical reviewers for each program
2. **Reviewer Sign-off**: Each reviewer to sign off on their respective program's proposed data model structure and open questions
3. **Scope Decision**: Facility management to decide on scope (all three vs facility-relevant subset)
4. **Integration Timing Decision**: Facility management to decide on DHIS2 integration timing
5. **Implementation Begin**: Once decisions are recorded in DECISIONS_PENDING.md item 11, begin implementation per program in separate branches or sequentially

---
*Prepared for Phase 7 National Disease Program Modules development*
*Reference: MCH/ANC module (`departments/mch/`) as structural reference for depth and complexity*