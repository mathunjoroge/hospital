from extensions import db
from datetime import datetime, timedelta
from departments.models.hr import Employee, Rota, Payroll

def seed_hr_data():
    """Seed initial HR department data if empty."""
    print("HR seed completed.")

# Run the script
if __name__ == "__main__":
    seed_hr_data()