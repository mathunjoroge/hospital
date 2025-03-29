from ..models import db

class DepartmentData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.Text, nullable=False)