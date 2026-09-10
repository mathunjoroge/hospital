# Prompt 8 of 9 — PACS / DICOMweb Integration (Phase 5)

Priority: Medium. DICOM metadata capture already exists; this phase adds the actual image
store/retrieve/display layer radiologists need.

Branch: `feat/phase5-pacs-dicomweb`. Never commit to `main` directly, never force-push. Run
the full test suite before starting and record the baseline.

| Item | Acceptance criteria |
|------|---------------------|
| P5-01 | Deploy Orthanc (open-source PACS) as a Docker container with a persistent volume. Configure DICOMweb (WADO-RS, STOW-RS, QIDO-RS). Add to `docker-compose.yml`. |
| P5-02 | Integrate OHIF Viewer v3 into the imaging department UI, loading studies from Orthanc via WADO-RS. |
| P5-03 | Wire imaging order to PACS: on study completion, write the Orthanc study UID to `RequestedImage.orthanc_uid`. Add a "View images" link opening OHIF. |
| P5-04 | Configure DICOM C-STORE from hospital CT/X-ray modalities to the Orthanc AE title. Test against each modality's actual DICOM conformance statement — don't assume WADO-RS is available on older CR/DR machines; confirm per modality (Orthanc bridges both C-STORE and DICOMweb). |
| P5-05 | Implement FHIR R4 `ImagingStudy`: maps Orthanc study/series/instance UIDs to the FHIR structure. Expose via `/api/fhir/R4/ImagingStudy`. |
| P5-06 | Migrate existing DICOM metadata rows to include `orthanc_uid`. Backfill historical studies loaded into Orthanc. |
| P5-07 | Configure Orthanc storage to the object storage backend resolved in `DECISIONS_PENDING.md` item 7 (MinIO/S3). 10-year retention. If item 7 isn't resolved, stop and ask rather than picking a backend. |
| P5-08 | Radiologist acceptance test: CT chest, MRI brain, and chest X-ray series display correctly in OHIF with measurements, windowing, and multi-planar reconstruction. This needs an actual radiologist or QA sign-off, not just an automated test — flag it in the PR if that review hasn't happened. |

## Risks to handle explicitly

- Studies can be 500MB+. Use JPEG-LS lossless transfer syntax for lossy-acceptable views;
  workflows requiring lossless imaging need dedicated workstations, not the standard viewer
  path — don't assume one config serves both cases.

## Done when

- Full test suite passes; ruff clean.
- P5-08's radiologist acceptance test is either completed or explicitly flagged as pending
  human review in the PR.
- Branch pushed (not force-pushed), ready for PR.
