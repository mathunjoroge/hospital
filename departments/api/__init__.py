from flask import Blueprint

bp = Blueprint('api', __name__)  # ✅ Register API Blueprint

from . import patients  # ✅ Import API routes

