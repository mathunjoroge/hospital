from flask import Blueprint

bp = Blueprint("api", __name__)  # ✅ Register API Blueprint

from . import (  # noqa: F401
    auth,  # ✅ JWT auth routes
    patients,  # ✅ Import API routes
)
