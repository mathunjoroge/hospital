from flask import Blueprint

bp = Blueprint('nursing', __name__, template_folder='templates')

from . import vitals, notes, care_tasks

