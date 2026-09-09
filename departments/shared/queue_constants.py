"""
Queue status constants for PatientWaitingList and appointment/patient flow tracking.
"""

class QueueStatus:
    NOT_SEEN = 0          # Default / portal
    SEEN_BILLED = 1       # Billing complete
    WAITING_TRIAGE = 4    # Registered, waiting for nursing (magic seen=4)
    VITALS_DONE = 5       # Vitals recorded by nursing, ready for doctor
    IN_CONSULTATION = 6   # Doctor has opened consultation/SOAP
    DISCHARGED = 7
    AWAITING_RESULTS = 8   # Consult done; patient at lab/imaging, may return
    AWAITING_PHARMACY = 9  # Cleared to collect drugs
    AWAITING_BILLING = 10  # Services done; awaiting settlement before exit        # Doctor consultation completed
