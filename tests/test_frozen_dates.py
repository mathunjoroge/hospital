from departments.forms import OncologyNoteForm
from departments.models.medicine import OncologyNote
from departments.models.records import ClinicBooking


def test_frozen_dates():
    """Verify that date fields are not frozen at import time."""
    assert callable(ClinicBooking.created_on.default.arg)
    assert callable(OncologyNote.note_date.default.arg)

    # Check the default on the UnboundField
    default = OncologyNoteForm.note_date.kwargs.get("default")
    assert callable(default)
