# Web Accessibility (WCAG 2.1 AA) Baseline & Audit Report

## 1. Overview & Disclaimer

Per Process Integrity rules (P.3), automated scans provide a baseline check for mechanical compliance (image ALT attributes, form labels, landmarks), but **do not replace a comprehensive manual accessibility review by human assistive technology specialists**.

---

## 2. Automated Inspection Findings & Remediation

| Page / Template Category | Element Scanned | Issue Identified | Remediation Applied | Status |
|--------------------------|-----------------|------------------|---------------------|--------|
| **Patient Portal Login** | `<form>` controls | Missing explicit `<label for="">` linkages. | Associated input IDs with `<label>` tags. | ✅ Fixed |
| **Analytics Dashboard** | Chart canvas | Missing fallback `aria-label` description. | Added `aria-label="Bed Occupancy and Revenue Trends Chart"`. | ✅ Fixed |
| **Break-Glass Audit Table** | `<table>` elements | Missing `<caption>` and `scope="col"` headers. | Added semantic table header scopes. | ✅ Fixed |
| **Color Contrast** | Secondary buttons | Contrast ratio fell below 4.5:1 on dark mode. | Updated CSS variable `--btn-secondary-bg` to meet 4.5:1 ratio. | ✅ Fixed |

---

## 3. Manual Assessment Requirements for Third-Party Reviewers

An expert accessibility review should evaluate the following manual user flows with screen readers (NVDA, JAWS, VoiceOver):

1. **Keyboard Navigation & Focus Traps**:
   - Navigation through consultation room WebRTC UI and complex modal dialogs using `Tab`, `Shift+Tab`, and `Escape` keys.
2. **Screen Reader Live Announcements**:
   - Verification that dynamic alerts (e.g. panic lab results or break-glass warnings) utilize `aria-live="assertive"` regions.
3. **High Contrast & Zoom Scaling**:
   - Testing UI legibility at 200% page zoom without horizontal scrolling or text overlap.
