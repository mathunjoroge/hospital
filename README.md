# Hospital Information Management System (HIMS)

## Overview
The **Hospital Information Management System (HIMS)** is a comprehensive solution designed to streamline and manage operations in a hospital setting. It provides functionality for patient registration, clinic management, nursing services, doctor services, pharmacy operations, laboratory management, cashiering, and payroll. User access and module loading are determined by the user type, ensuring that staff members only interact with the modules relevant to their roles.

---

## Features
### 1. **Patient Registration**
   - Register new patients with their personal details, contact information, and medical history.
   - Manage and update existing patient records.

### 2. **Clinic Management**
   - Schedule and manage clinic appointments.
   - Assign patients to clinics based on their needs.

### 3. **Nursing Services**
   - Maintain records of patient care.
   - Manage nursing schedules and assigned tasks.

### 4. **Doctor Services**
   - Assign doctors to patients.
   - Record diagnoses and prescribed treatments.
   - View and manage patient histories.

### 5. **Pharmacy**
   - Manage drug inventory.
   - Issue prescribed medications to patients.
   - Track drug usage and reorder stock.

### 6. **Laboratory**
   - Manage lab test requests and results.
   - Store and retrieve lab reports.

### 7. **Cashier**
   - Handle patient billing and payments.
   - Generate invoices and receipts.

### 8. **Payroll**
   - Manage staff salary records.
   - Generate payslips and reports.

---

## User Roles and Access Control
The system supports role-based access to modules:

- **Admin**: Full access to all modules and system settings.
- **Registration**: Access to patient registration and clinic management.
- **Nurse**: Access to nursing services and patient records.
- **Doctor**: Access to patient records, doctor services, and lab results.
- **Pharmacist**: Access to the pharmacy module.
- **Lab Technician**: Access to laboratory management.
- **Cashier**: Access to billing and payments.
- **HR**: Access to payroll.

---

## Technologies Used
- **Programming Language**: PHP
- **Database**: MySQL
- **Frontend**: HTML, CSS, JavaScript
- **Server**: Apache

---

## Installation Guide
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/hospital-management-system.git
   ```

2. **Set Up the Database**:
   - Create a MySQL database (e.g., `hospita`).
   - Import the provided `hospital.sql` file into the database.

3. **Configure Database Connection**:
change the details in connect.php to your credentials
     ```

4. **Deploy the Application**:
   - Place the project files in your web server's root directory (e.g., `htdocs` for XAMPP).
   - Start your Apache and MySQL services.
   - Access the system via `http://localhost/hospital-management-system`.

---

## Usage
1. Log in using your credentials.
2. Depending on your user type, the corresponding module(s) will be displayed.
3. Navigate through the system using the menu to perform your tasks.

---


## Contribution Guidelines
1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Commit your changes and push them to your branch.
4. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.



