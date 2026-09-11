from flask import Blueprint

# Create the blueprint
bp = Blueprint("billing", __name__, template_folder="templates")

# Import routes to register them with the blueprint
import departments.billing.event_listeners  # noqa: F401

from . import routes  # noqa: F401
