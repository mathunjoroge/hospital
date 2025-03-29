from extensions import db
from datetime import datetime

class Log(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    level = db.Column(db.String(20), nullable=False)  # e.g., INFO, ERROR, WARNING
    message = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    source = db.Column(db.String(50))  # e.g., 'auth', 'pharmacy'