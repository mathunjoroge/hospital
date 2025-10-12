from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import os

def create_lab_report_pdf(filename="sample_lab_report.pdf"):
    # Create a PDF canvas
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    margin = 0.5 * inch
    y_position = height - margin

    # Helper function to draw text
    def draw_text(text, x, y, font="Helvetica", size=12, bold=False):
        c.setFont(f"{font}-{'Bold' if bold else 'Oblique'}" if bold else font, size)
        c.drawString(x, y, text)
        return y - size * 1.2

    # Header: Lab Information
    c.setFillColor(colors.black)
    y_position = draw_text("HMIS Clinical Laboratory", margin, y_position, size=16, bold=True)
    y_position = draw_text("123 Medical Way, Health City, HC 12345", margin, y_position, size=10)
    y_position = draw_text("Phone: (123) 456-7890 | Email: lab@hmis.org", margin, y_position, size=10)
    y_position = draw_text(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", margin, y_position, size=10)
    y_position -= 10

    # Patient Information
    y_position = draw_text("Patient Information", margin, y_position, size=14, bold=True)
    y_position = draw_text("Name: John Doe", margin, y_position)
    y_position = draw_text("Patient ID: JD123456", margin, y_position)
    y_position = draw_text("DOB: 1985-04-15", margin, y_position)
    y_position = draw_text("Gender: Male", margin, y_position)
    y_position -= 10

    # Test Results: Complete Blood Count (CBC)
    y_position = draw_text("Complete Blood Count (CBC)", margin, y_position, size=14, bold=True)
    y_position = draw_text("Test Name                Result          Units          Reference Range", margin, y_position, size=10, bold=True)
    y_position = draw_text("-" * 80, margin, y_position, size=10)
    cbc_tests = [
        ("WBC", "6.5", "x10^3/uL", "4.0 - 11.0"),
        ("RBC", "4.8", "x10^6/uL", "4.5 - 5.9"),
        ("Hemoglobin", "14.2", "g/dL", "13.5 - 17.5"),
        ("Hematocrit", "42.0", "%", "38.0 - 50.0"),
        ("Platelets", "250", "x10^3/uL", "150 - 450")
    ]
    for test, result, unit, ref in cbc_tests:
        y_position = draw_text(f"{test:<20} {result:<15} {unit:<15} {ref}", margin, y_position, size=10)
    y_position -= 10

    # Test Results: Comprehensive Metabolic Panel (CMP)
    y_position = draw_text("Comprehensive Metabolic Panel (CMP)", margin, y_position, size=14, bold=True)
    y_position = draw_text("Test Name                Result          Units          Reference Range", margin, y_position, size=10, bold=True)
    y_position = draw_text("-" * 80, margin, y_position, size=10)
    cmp_tests = [
        ("Glucose", "110", "mg/dL", "70 - 99"),
        ("BUN", "15", "mg/dL", "7 - 20"),
        ("Creatinine", "0.9", "mg/dL", "0.7 - 1.3"),
        ("Sodium", "140", "mmol/L", "135 - 145"),
        ("Potassium", "4.2", "mmol/L", "3.5 - 5.0"),
        ("HbA1c", "6.8", "%", "4.0 - 5.6")
    ]
    for test, result, unit, ref in cmp_tests:
        y_position = draw_text(f"{test:<20} {result:<15} {unit:<15} {ref}", margin, y_position, size=10)
    y_position -= 10

    # Notes
    y_position = draw_text("Notes", margin, y_position, size=14, bold=True)
    y_position = draw_text("Elevated HbA1c suggests prediabetes. Recommend follow-up with endocrinologist.", margin, y_position, size=10)
    y_position = draw_text("All other results within normal limits.", margin, y_position, size=10)

    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(margin, margin, "Confidential: For authorized medical use only.")
    
    # Save the PDF
    c.save()
    print(f"Lab report PDF created: {filename}")

if __name__ == "__main__":
    create_lab_report_pdf()