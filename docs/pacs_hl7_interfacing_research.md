# HL7 v2 Lab-Instrument Interfacing & PACS/DICOM Architectural Recommendation

## Executive Summary

This evaluation addresses two critical infrastructure capabilities required for clinical operations at scale:
1. **Lab-Instrument Interfacing**: Bidirectional connectivity with clinical chemistry and hematology analyzers.
2. **Medical Imaging / PACS Integration**: Storage, retrieval, and browser-based rendering of DICOM imaging studies (X-ray, CT, Ultrasound).

---

## 1. HL7 v2 Laboratory Instrument Interfacing

### 1.1 Requirements & Standards
Most clinical laboratory analyzers (e.g. Sysmex, Mindray, Roche Cobas, Abbott Architect) communicate over Serial (RS-232) or TCP/IP using ASTM E1381/E1394 or HL7 v2.x (v2.3.1 / v2.5.1) framing standards via MLLP (Minimal Lower Layer Protocol).

Key message types required:
* `ORM^O01`: Order Messages (HMIS -> LIS / Instrument) containing test orders and patient demographic IDs.
* `ORU^R01`: Observation Results (Instrument / LIS -> HMIS) containing numerical/text results, units, reference ranges, and flag statuses (High/Low/Panic).
* `ACK^R01`: Application Acknowledgement message indicating successful database persistence.

### 1.2 Architectural Design for Flask HMIS
Directly connecting Flask HTTP worker processes to raw TCP sockets or RS-232 ports is strongly discouraged due to thread blocking and event loop incompatibility.

Recommended Topology:
```
+------------------+         ASTM/MLLP TCP        +----------------------+
| Lab Analyzers    | <--------------------------> | MLLP Receiver Daemon |
| (Sysmex/Roche)   |                              | (Python Asyncio/HL7) |
+------------------+                              +----------+-----------+
                                                             | JSON HTTP/REST
                                                             v
+------------------+                              +----------------------+
| PostgreSQL DB    | <--------------------------- | Flask HMIS Internal  |
| (Laboratory)     |                              | Lab API Endpoint     |
+------------------+                              +----------------------+
```

1. **Ingest Gateway**: An isolated, lightweight Python daemon (`hl7apy` or `python-hl7`) listening on TCP Port 2575 (MLLP protocol).
2. **Payload Parsing**: Convert `ORU^R01` segments (`MSH`, `PID`, `OBR`, `OBX`) to structured JSON.
3. **Internal API Submission**: Post structured JSON payload to `/api/laboratory/results/ingest` with HMAC signature validation.
4. **Panic Result Dispatch**: If `OBX-8` contains critical flag (`HH`, `LL`, `AA`), trigger SMS/email alert to ordering physician immediately.

---

## 2. Medical Imaging: DICOM Viewer vs. Full PACS Integration

### 2.1 Full PACS (Picture Archiving and Communication System) Scope
A full PACS deployment (e.g., Orthanc DICOM server or dcm4chee) requires:
* High-availability DICOM C-STORE / C-FIND / C-MOVE listeners on Port 104.
* Large-scale SAN/NAS storage infrastructure for multi-gigabyte CT/MRI DICOM series.
* Complex DICOM router configuration and DICOM TLS encryption.
* VNA (Vendor Neutral Archive) compliance for multi-modality storage.

### 2.2 Near-Term Recommendation: Browser-Based DICOM Viewer (Cornerstone.js)

For this facility's immediate operational profile, deploying a full enterprise PACS is **unnecessary overhead**. Instead, a **lightweight, browser-based DICOM viewer powered by Cornerstone.js (or OHIF Viewer) integrated with an S3/MinIO DICOM Web (WADO-RS/STOW-RS) backend** is the recommended near-term solution.

#### Architecture:
1. **DICOM Web Gateway**: Deploy an open-source Orthanc DICOM server acting solely as a WADO-RS / DICOM Web proxy.
2. **Client Rendering**: Embed `Cornerstone.js` canvas into `departments/imaging/templates/imaging/view_study.html`.
3. **Features**: Pan, zoom, windowing (WW/WL), measurement tools (length, angle, HU units), and multi-frame playback.
4. **Access Control**: HMIS session-authenticated endpoint `/imaging/study/<study_uid>/wado` proxies requests to Orthanc after verifying user role (`doctor`, `radiologist`, `admin`).

---

## 3. Implementation Roadmap

| Phase | Milestone | Est. Effort | Target Outcome |
|-------|-----------|-------------|----------------|
| **Phase 1** | MLLP Receiver Daemon & Ingest API | 2 Weeks | Automated lab result population from automated analyzers. |
| **Phase 2** | Orthanc DICOM Web Server Dockerization | 1 Week | WADO-RS DICOM Web storage connected to S3/MinIO. |
| **Phase 3** | Cornerstone.js Frontend Integration | 1 Week | In-browser DICOM viewing within Patient Imaging records. |
