from flask import render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from ..models import MortuaryData
from . import bp

@bp.route('/')
@login_required
def index():
    if current_user.role != 'mortuary':
        return redirect(url_for('login'))
    return render_template('mortuary.html')