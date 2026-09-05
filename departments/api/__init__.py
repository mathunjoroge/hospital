from flask import Blueprint

bp = Blueprint('api', __name__)  # ✅ Register API Blueprint

from . import patients   # ✅ Import API routes
from . import auth       # ✅ JWT auth routes
