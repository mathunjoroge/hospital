"""
tests/test_hr_module.py
─────────────────────────
HR: employee creation/update, payroll generation, and self-service.

Zero tests existed for departments/hr before this file (27% live coverage on
a department with real routes). Three core routes crashed or silently failed
on every real submission:

  - new_employee(): Employee.generate_employee_id() was called but never
    defined, and the allowance/deduction assignment code referenced a
    per-employee schema (Employee.allowances relationship, Deduction.type,
    Deduction.employee_id) that doesn't exist on the real models.
  - update_employee(): the template's field was named "position"; the model
    has no such column and the route reads request.form.get("role"), which
    was therefore always None, so validation failed on every submission.
  - generate_payroll(): referenced employee.basic_salary, a column that
    didn't exist, so it raised AttributeError on the first employee, every
    time.

Separately, five self-service routes compared current_user.id (a User PK)
directly against Employee.id (an unrelated table's PK) as if they were the
same identifier, which only worked when the two happened to coincide.
"""

from datetime import datetime

import pytest
from werkzeug.security import generate_password_hash

from departments.models.hr import Allowance, AuditLog, Deduction, Employee, Leave, Payroll
from departments.models.user import User
from extensions import db

# ── P0 regressions: the profile/leave/delete crash cluster ────────────────

def test_employee_profile_renders_for_hr(hr_client, app):
    """
    Regression: the template called form.hidden_tag() but the HR route passed
    no form object, and it iterated employee.leaves which had no relationship
    — so HR's central profile page raised UndefinedError on every visit.
    """
    employee = _make_employee(name="Profile Target")
    response = hr_client.get(f"/hr/employee_profile/{employee.id}")
    assert response.status_code == 200
    assert b"Profile Target" in response.data


def test_employee_profile_shows_leave_history(client, app, linked_pair):
    """employee.leaves must resolve now that the relationship exists."""
    import datetime as dt

    employee, user = linked_pair
    db.session.add(Leave(employee_id=employee.id,
                         start_date=dt.datetime(2026, 10, 1),
                         end_date=dt.datetime(2026, 10, 5),
                         type="sick", status="Pending"))
    db.session.commit()

    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"},
                follow_redirects=True)
    response = client.get("/hr/update_employee_profile")  # self-service profile
    assert response.status_code == 200
    assert b"2026-10-01" in response.data


def test_approve_leave_does_not_crash_after_commit(hr_client, app):
    """
    Regression: send_email called Mail.send() on the flask_mail Mail CLASS
    (the app-bound instance was never reachable from hr.routes), so approval
    committed the status change and then raised TypeError — a 500 after a
    successful DB write. Exercises the real code path: an employee with no
    email on file (recipient=None used to raise inside flask_mail too).
    """
    employee = _make_employee(name="Leave Target")  # no email set
    leave = Leave(employee_id=employee.id,
                  start_date=datetime(2026, 10, 1), end_date=datetime(2026, 10, 5),
                  type="sick", status="Pending")
    db.session.add(leave)
    db.session.commit()

    response = hr_client.get(f"/hr/approve_leave/{leave.id}", follow_redirects=True)
    assert response.status_code == 200
    assert leave.status == "Approved"


def test_send_email_survives_missing_mail_extension(app):
    """current_app.extensions['mail'] absent must degrade, not raise."""
    from departments.hr.routes import send_email
    with app.app_context():
        had_mail = app.extensions.pop("mail", None)
        try:
            assert send_email("Subject", "hr@example.com", "body") is False
        finally:
            if had_mail is not None:
                app.extensions["mail"] = had_mail


def test_reject_leave_does_not_crash(hr_client, app):
    employee = _make_employee(name="Reject Target")
    leave = Leave(employee_id=employee.id,
                  start_date=datetime(2026, 10, 1), end_date=datetime(2026, 10, 5),
                  type="sick", status="Pending")
    db.session.add(leave)
    db.session.commit()
    response = hr_client.get(f"/hr/reject_leave/{leave.id}", follow_redirects=True)
    assert response.status_code == 200
    assert leave.status == "Rejected"


def test_send_email_skips_silently_without_recipient(app):
    """Unit-level: send_email(None) must not raise (previously recipients=[None])."""
    from departments.hr.routes import send_email
    with app.app_context():
        assert send_email("Subject", None, "body") is False


def test_update_profile_rejects_blank_name(hr_client, app):
    """
    Regression: update_profile wrote request.form values straight to
    nullable=False columns, so a blank name raised IntegrityError at commit
    and 500'd instead of flashing a validation error.
    """
    employee = _make_employee(name="Original Name")
    response = hr_client.post(f"/hr/update_profile/{employee.id}", data={
        "name": "", "department": "nursing", "job_group": "Group A",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert db.session.get(Employee, employee.id).name == "Original Name"


def test_update_profile_updates_and_stamps_updated_by(hr_client, app):
    employee = _make_employee(name="Before")
    assert employee.updated_by is None
    hr_client.post(f"/hr/update_profile/{employee.id}", data={
        "name": "After", "department": "records", "job_group": "Group B",
    })
    updated = db.session.get(Employee, employee.id)
    assert updated.name == "After"
    assert updated.department == "records"
    assert updated.job_group == "Group B"
    assert updated.updated_by == User.query.filter_by(username="hr_clerk").first().id


def test_employee_cannot_update_another_employees_profile(client, app):
    """The POST-only route is shared by self-service; cross-employee writes must 302 away."""
    db.session.add(User(username="plain_emp", role="nursing",
                        password=generate_password_hash("Plain!234")))
    db.session.commit()
    other = _make_employee(employee_id="E-OTHER2", name="Other Person")
    client.post("/login", data={"username": "plain_emp", "password": "Plain!234"},
                follow_redirects=True)
    client.post(f"/hr/update_profile/{other.id}", data={
        "name": "Hacked Name", "department": "nursing", "job_group": "Group A",
    }, follow_redirects=False)
    assert db.session.get(Employee, other.id).name == "Other Person"


def test_linked_employee_can_update_own_profile_via_update_profile(client, app, linked_pair):
    employee, user = linked_pair
    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"},
                follow_redirects=True)
    response = client.post(f"/hr/update_profile/{employee.id}", data={
        "name": employee.name, "department": "nursing", "job_group": "Group A",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert db.session.get(Employee, employee.id).name == "Linked Nurse"


def test_delete_employee_with_history_is_deactivated_not_deleted(hr_client, app):
    """
    Regression: payrolls.employee_id has no ON DELETE CASCADE, so DELETE
    raised IntegrityError for anyone with payroll history and the route
    showed a generic error while the row stayed. Deactivate instead.
    """
    employee = _make_employee(employee_id="E-HIST", name="History Holder")
    db.session.add(Payroll(employee_id=employee.id, month="2026-09",
                           gross_pay=100, total_deductions=0, net_pay=100))
    db.session.commit()
    employee_pk = employee.id

    response = hr_client.post(f"/hr/delete_employee/{employee_pk}", follow_redirects=True)
    assert response.status_code == 200
    survivor = db.session.get(Employee, employee_pk)
    assert survivor is not None  # row retained for statutory/audit integrity
    assert survivor.is_active is False


def test_delete_employee_without_history_still_hard_deletes(hr_client, app):
    employee = _make_employee(employee_id="E-FRESH", name="Fresh Hire No History")
    employee_pk = employee.id
    hr_client.post(f"/hr/delete_employee/{employee_pk}", follow_redirects=True)
    assert db.session.get(Employee, employee_pk) is None


def test_leave_request_success_redirects_employees_away_from_403(client, app, linked_pair):
    """
    Regression: success redirected to hr.index (roles_required hr/admin), so
    a regular employee got a 403 immediately after filing a request.
    """
    employee, user = linked_pair
    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"},
                follow_redirects=True)
    response = client.post("/hr/leave_request", data={
        "start_date": "2026-10-01", "end_date": "2026-10-05", "type": "vacation",
    }, follow_redirects=False)
    assert response.status_code == 302
    # Must not land on an HR-only endpoint.
    assert "/hr/index" not in response.headers["Location"]
    follow = client.get(response.headers["Location"], follow_redirects=False)
    assert follow.status_code == 200


@pytest.fixture
def hr_client(client, app):
    db.session.add(User(username="hr_clerk", role="hr",
                        password=generate_password_hash("Clerk!2345")))
    db.session.commit()
    client.post("/login", data={"username": "hr_clerk", "password": "Clerk!2345"},
                follow_redirects=True)
    return client


def _make_employee(**overrides):
    defaults = dict(employee_id="E-BASE", name="Base Employee", role="nursing",
                    department="nursing", job_group="Group A")
    defaults.update(overrides)
    emp = Employee(**defaults)
    db.session.add(emp)
    db.session.commit()
    return emp


# ── new_employee: the crash on every submission ───────────────────────────

def test_new_employee_no_longer_crashes(hr_client, app):
    """Regression: Employee.generate_employee_id() didn't exist; every POST 500'd."""
    response = hr_client.post("/hr/new_employee", data={
        "name": "Jane Recruit", "role": "nursing", "department": "nursing",
        "job_group": "Group A",
    }, follow_redirects=True)
    assert response.status_code == 200
    employee = Employee.query.filter_by(name="Jane Recruit").first()
    assert employee is not None
    assert employee.employee_id.startswith("E")


def test_generated_employee_ids_are_sequential(hr_client, app):
    for name in ("First Hire", "Second Hire"):
        hr_client.post("/hr/new_employee", data={
            "name": name, "role": "nursing", "department": "nursing",
            "job_group": "Group A",
        })
    ids = sorted(e.employee_id for e in Employee.query.all())
    assert ids[0] < ids[1]
    assert len(set(ids)) == len(ids)


def test_new_employee_accepts_basic_salary(hr_client, app):
    hr_client.post("/hr/new_employee", data={
        "name": "Salaried Hire", "role": "nursing", "department": "nursing",
        "job_group": "Group A", "basic_salary": "45000",
    })
    employee = Employee.query.filter_by(name="Salaried Hire").first()
    assert float(employee.basic_salary) == 45000.0


def test_new_employee_without_salary_is_valid(hr_client, app):
    """A new hire with no salary set yet is a valid state, not an error."""
    response = hr_client.post("/hr/new_employee", data={
        "name": "Unsalaried Hire", "role": "nursing", "department": "nursing",
        "job_group": "Group A",
    }, follow_redirects=True)
    assert response.status_code == 200
    employee = Employee.query.filter_by(name="Unsalaried Hire").first()
    assert employee.basic_salary is None


def test_new_employee_rejects_negative_salary(hr_client, app):
    hr_client.post("/hr/new_employee", data={
        "name": "Bad Salary Hire", "role": "nursing", "department": "nursing",
        "job_group": "Group A", "basic_salary": "-500",
    })
    assert Employee.query.filter_by(name="Bad Salary Hire").first() is None


def test_new_employee_does_not_reference_dead_schema(hr_client, app):
    """
    Regression: the old implementation called new_employee.allowances.extend(...)
    (no such relationship) and created Deduction(employee_id=..., type=...)
    (no such columns) unconditionally on every submission.
    """
    # A global Allowance/Deduction existing must not make creation crash.
    db.session.add(Allowance(job_group="Group A", name="Housing", value=5000))
    db.session.add(Deduction(name="PAYE", value=10, is_percentage=True))
    db.session.commit()

    response = hr_client.post("/hr/new_employee", data={
        "name": "Post Schema Fix", "role": "nursing", "department": "nursing",
        "job_group": "Group A",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert Employee.query.filter_by(name="Post Schema Fix").first() is not None
    # No stray per-employee Deduction row should have been fabricated.
    assert Deduction.query.filter_by(name="PAYE").count() == 1


def test_new_employee_requires_all_fields(hr_client, app):
    hr_client.post("/hr/new_employee", data={"name": "Incomplete"})
    assert Employee.query.filter_by(name="Incomplete").first() is None


def test_new_employee_form_get_still_renders(hr_client, app):
    """The GET form must render even though allowances/deductions display changed."""
    db.session.add(Allowance(job_group="Group A", name="Housing", value=5000))
    db.session.add(Deduction(name="NHIF", value=500, is_percentage=False))
    db.session.commit()
    response = hr_client.get("/hr/new_employee")
    assert response.status_code == 200
    assert b"Housing" in response.data
    assert b"NHIF" in response.data


# ── update_employee: the position/role field mismatch ─────────────────────

def test_update_employee_no_longer_silently_fails(hr_client, app):
    """
    Regression: the form field was 'position', the model/route expected
    'role', so request.form.get('role') was always None and every
    submission raised 'All fields are required!' without changing anything.
    """
    employee = _make_employee(name="Old Name")
    response = hr_client.post(f"/hr/update_employee/{employee.id}", data={
        "name": "New Name", "role": "pharmacy", "department": "pharmacy",
        "is_active": "on",
    }, follow_redirects=True)
    assert response.status_code == 200
    updated = db.session.get(Employee, employee.id)
    assert updated.name == "New Name"
    assert updated.role == "pharmacy"


def test_update_employee_form_uses_role_not_position(app, hr_client):
    """The rendered form must post a field named 'role', matching the model."""
    employee = _make_employee()
    response = hr_client.get(f"/hr/update_employee/{employee.id}")
    body = response.get_data(as_text=True)
    assert 'name="role"' in body
    assert 'name="position"' not in body


def test_update_employee_can_set_basic_salary(hr_client, app):
    employee = _make_employee()
    hr_client.post(f"/hr/update_employee/{employee.id}", data={
        "name": employee.name, "role": employee.role, "department": employee.department,
        "basic_salary": "60000",
    })
    assert float(db.session.get(Employee, employee.id).basic_salary) == 60000.0


def test_update_employee_preserves_salary_when_field_left_blank(hr_client, app):
    employee = _make_employee(basic_salary=50000)
    hr_client.post(f"/hr/update_employee/{employee.id}", data={
        "name": employee.name, "role": employee.role, "department": employee.department,
    })
    assert float(db.session.get(Employee, employee.id).basic_salary) == 50000.0


# ── generate_payroll: the basic_salary crash ──────────────────────────────

def test_generate_payroll_no_longer_crashes(hr_client, app):
    """Regression: employee.basic_salary didn't exist; this 500'd on the first employee."""
    _make_employee(employee_id="E-PR1", name="Payroll Target", basic_salary=50000)
    response = hr_client.get("/hr/generate_payroll/2026-09", follow_redirects=True)
    assert response.status_code == 200
    assert Payroll.query.filter_by(month="2026-09").count() == 1


def test_generate_payroll_skips_employees_without_salary(hr_client, app):
    """An employee with no basic_salary must be skipped, not crash the whole run."""
    _make_employee(employee_id="E-NOSAL", name="No Salary Yet")
    _make_employee(employee_id="E-HASSAL", name="Has Salary", basic_salary=40000)

    response = hr_client.get("/hr/generate_payroll/2026-09", follow_redirects=True)
    assert response.status_code == 200
    assert Payroll.query.filter_by(month="2026-09").count() == 1
    paid = Payroll.query.filter_by(month="2026-09").first()
    assert paid.employee.name == "Has Salary"
    assert b"No Salary Yet" in response.data  # flashed as skipped


def test_generate_payroll_applies_allowances_and_deductions(hr_client, app):
    _make_employee(employee_id="E-CALC", name="Calc Target",
                   job_group="Group A", basic_salary=10000)
    db.session.add(Allowance(job_group="Group A", name="Housing", value=2000))
    db.session.add(Deduction(name="PAYE", value=10, is_percentage=True))
    db.session.commit()

    hr_client.get("/hr/generate_payroll/2026-09")
    payroll = Payroll.query.filter_by(month="2026-09").first()

    # gross = 10000 + 2000 = 12000; deduction = 10% of 12000 = 1200; net = 10800
    assert float(payroll.gross_pay) == 12000.0
    assert float(payroll.total_deductions) == 1200.0
    assert float(payroll.net_pay) == 10800.0


def test_generate_payroll_is_reachable_from_the_dashboard(hr_client, app):
    """Regression: no template ever linked to generate_payroll at all."""
    response = hr_client.get("/hr/payroll")
    assert b"generate_payroll" in response.data or b"Generate" in response.data


# ── Employee <-> User identity link ───────────────────────────────────────

@pytest.fixture
def linked_pair(app):
    """A User and an Employee, deliberately created so their IDs diverge."""
    for i in range(3):
        db.session.add(User(username=f"filler{i}", role="nursing",
                            password=generate_password_hash("Filler!234")))
    db.session.commit()
    employee = Employee(employee_id="E-LINKED", name="Linked Nurse", role="nursing",
                        department="nursing", job_group="Group A")
    db.session.add(employee)
    user = User(username="linked_nurse", role="nursing",
               password=generate_password_hash("Nurse!2345"))
    db.session.add(user)
    db.session.commit()
    employee.user_id = user.id
    db.session.commit()
    assert employee.id != user.id, "test setup must not coincidentally collide"
    return employee, user


def test_self_service_payslips_visible_when_linked(client, app, linked_pair):
    """Regression: this always came back empty because of the ID mismatch."""
    employee, user = linked_pair
    db.session.add(Payroll(employee_id=employee.id, month="2026-09",
                           gross_pay=50000, total_deductions=5000, net_pay=45000))
    db.session.commit()

    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"},
                follow_redirects=True)
    response = client.get("/hr/payslips")
    assert response.status_code == 200
    assert b"45000" in response.data or b"45000.00" in response.data


def test_self_service_view_own_payslip_when_linked(client, app, linked_pair):
    """Regression: viewing your own payslip 302'd away as 'unauthorized'."""
    employee, user = linked_pair
    payroll = Payroll(employee_id=employee.id, month="2026-09",
                      gross_pay=50000, total_deductions=5000, net_pay=45000)
    db.session.add(payroll)
    db.session.commit()
    payroll_id = payroll.id

    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"},
                follow_redirects=True)
    response = client.get(f"/hr/view_payslip/{payroll_id}")
    assert response.status_code == 200


def test_id_collision_no_longer_grants_cross_access(client, app):
    """
    Regression: the old check was `payroll.employee_id != current_user.id`.
    If a User's PK happened to equal a *different* employee's PK, that user
    could view the other employee's payslip. Deliberately construct that
    coincidence and confirm the fixed check isn't fooled by it.
    """
    victim = Employee(employee_id="E-VICTIM", name="Victim", role="nursing",
                      department="nursing", job_group="Group A")
    db.session.add(victim)
    db.session.commit()
    victim_id = victim.id

    # Manufacture an attacker User whose PK equals the victim Employee's PK.
    attacker = User(username="attacker", role="nursing",
                    password=generate_password_hash("Attack!234"))
    db.session.add(attacker)
    db.session.commit()
    while attacker.id != victim_id:
        db.session.delete(attacker)
        db.session.commit()
        attacker = User(username=f"attacker{victim_id}", role="nursing",
                        password=generate_password_hash("Attack!234"))
        db.session.add(attacker)
        db.session.commit()
    assert attacker.id == victim_id  # the coincidence the old code trusted

    payroll = Payroll(employee_id=victim_id, month="2026-09",
                      gross_pay=99999, total_deductions=0, net_pay=99999)
    db.session.add(payroll)
    db.session.commit()
    payroll_id, attacker_username = payroll.id, attacker.username

    client.post("/login", data={"username": attacker_username, "password": "Attack!234"},
                follow_redirects=True)
    response = client.get(f"/hr/view_payslip/{payroll_id}", follow_redirects=False)
    assert response.status_code == 302  # denied, despite the ID coincidence


def test_unlinked_employee_gets_empty_payslips_not_someone_elses(client, app):
    db.session.add(User(username="unlinked_user", role="nursing",
                        password=generate_password_hash("Unlinked!23")))
    other = Employee(employee_id="E-OTHER", name="Other Employee", role="nursing",
                     department="nursing", job_group="Group A")
    db.session.add(other)
    db.session.commit()
    db.session.add(Payroll(employee_id=other.id, month="2026-09",
                           gross_pay=1, total_deductions=0, net_pay=1))
    db.session.commit()

    client.post("/login", data={"username": "unlinked_user", "password": "Unlinked!23"},
                follow_redirects=True)
    response = client.get("/hr/payslips")
    assert response.status_code == 200
    assert b"Other Employee" not in response.data


def test_leave_request_uses_linked_employee(client, app, linked_pair):
    """Regression: Leave.employee_id was set to current_user.id (a User PK)."""
    employee, user = linked_pair
    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"},
                follow_redirects=True)
    client.post("/hr/leave_request", data={
        "start_date": "2026-10-01", "end_date": "2026-10-05", "type": "vacation",
    }, follow_redirects=True)

    leave = Leave.query.first()
    assert leave is not None
    assert leave.employee_id == employee.id


def test_leave_request_without_link_is_refused_not_miswritten(client, app):
    """
    An unlinked account must not be able to file a leave request under the
    wrong employee_id (or under current_user.id as if it were one).
    """
    db.session.add(User(username="no_link_user", role="nursing",
                        password=generate_password_hash("NoLink!234")))
    db.session.commit()
    client.post("/login", data={"username": "no_link_user", "password": "NoLink!234"},
                follow_redirects=True)
    response = client.post("/hr/leave_request", data={
        "start_date": "2026-10-01", "end_date": "2026-10-05", "type": "sick",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert Leave.query.count() == 0


def test_hr_can_link_employee_to_login_account(hr_client, app):
    employee = _make_employee(name="To Be Linked")
    db.session.add(User(username="future_login", role="nursing",
                        password=generate_password_hash("Future!234")))
    db.session.commit()

    hr_client.post(f"/hr/update_employee/{employee.id}", data={
        "name": employee.name, "role": employee.role, "department": employee.department,
        "linked_username": "future_login",
    })
    updated = db.session.get(Employee, employee.id)
    assert updated.user.username == "future_login"


def test_cannot_link_one_account_to_two_employees(hr_client, app):
    shared_user = User(username="shared_login", role="nursing",
                       password=generate_password_hash("Shared!234"))
    db.session.add(shared_user)
    emp_a = _make_employee(employee_id="E-A", name="Employee A")
    emp_b = _make_employee(employee_id="E-B", name="Employee B")
    db.session.commit()

    hr_client.post(f"/hr/update_employee/{emp_a.id}", data={
        "name": emp_a.name, "role": emp_a.role, "department": emp_a.department,
        "linked_username": "shared_login",
    })
    hr_client.post(f"/hr/update_employee/{emp_b.id}", data={
        "name": emp_b.name, "role": emp_b.role, "department": emp_b.department,
        "linked_username": "shared_login",
    })

    assert db.session.get(Employee, emp_a.id).user_id == shared_user.id
    assert db.session.get(Employee, emp_b.id).user_id != shared_user.id


def test_linking_unknown_username_is_rejected(hr_client, app):
    employee = _make_employee()
    response = hr_client.post(f"/hr/update_employee/{employee.id}", data={
        "name": employee.name, "role": employee.role, "department": employee.department,
        "linked_username": "does_not_exist",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert db.session.get(Employee, employee.id).user_id is None


# ── The duplicate get_effective_role() ────────────────────────────────────

def test_hr_uses_the_shared_get_effective_role():
    """Regression: hr/routes.py defined its own divergent copy of this."""
    import departments.hr.routes as hr_routes
    import departments.rbac as rbac
    assert hr_routes.get_effective_role is rbac.get_effective_role


# ── Dead template cleanup ──────────────────────────────────────────────────

def test_dead_payroll_templates_are_removed():
    """
    These seven templates referenced a Salary model, an Employee.allowances
    relationship, and a per-employee Deduction schema that never existed, had
    zero incoming links from anywhere in the app, and were never rendered by
    any route.
    """
    import os
    for name in ("salaries", "new_salary", "deductions", "new_deduction",
                 "generate_payslip", "payslips", "view_payslip"):
        path = f"departments/hr/templates/hr/{name}.html"
        assert not os.path.exists(path), f"{path} should have been removed"


def test_hr_templates_reference_no_missing_endpoints(app):
    """Guard against another round of the position/role-style mismatch."""
    import re
    from pathlib import Path

    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}
    pattern = re.compile(r"url_for\(\s*['\"]([a-z_]+\.[a-z_]+)['\"]", re.IGNORECASE)

    broken = []
    for template in Path("departments/hr").rglob("*.html"):
        for endpoint in pattern.findall(template.read_text(errors="replace")):
            if endpoint not in endpoints:
                broken.append(f"{template}: {endpoint}")

    assert not broken, "HR templates reference endpoints that do not exist:\n" + "\n".join(broken)


# ── HR audit trail: previously nothing ever wrote to AuditLog ─────────────

def test_new_employee_writes_an_audit_log_entry(hr_client, app):
    hr_client.post("/hr/new_employee", data={
        "name": "Audited Hire", "role": "nursing", "department": "nursing",
        "job_group": "Group A",
    })
    log = AuditLog.query.order_by(AuditLog.id.desc()).first()
    assert log is not None
    assert log.action == "Employee Created"
    assert "Audited Hire" in log.details


def test_delete_employee_writes_an_audit_log_entry(hr_client, app):
    employee = _make_employee(employee_id="E-DEL", name="Delete Target")
    employee_pk = employee.id
    hr_client.post(f"/hr/delete_employee/{employee_pk}")
    log = AuditLog.query.filter_by(action="Employee Deleted").first()
    assert log is not None
    assert "Delete Target" in log.details


def test_approve_leave_writes_an_audit_log_entry(hr_client, app):
    employee = _make_employee(employee_id="E-APR", name="Approve Target")
    leave = Leave(employee_id=employee.id,
                  start_date=datetime(2026, 10, 1), end_date=datetime(2026, 10, 2),
                  type="vacation", status="Pending")
    db.session.add(leave)
    db.session.commit()

    hr_client.get(f"/hr/approve_leave/{leave.id}")
    log = AuditLog.query.filter_by(action="Leave Approved").first()
    assert log is not None
    assert "Approve Target" in log.details


def test_add_allowance_and_deduction_write_audit_log_entries(hr_client, app):
    hr_client.post("/hr/add_allowance", data={
        "job_group": "Group A", "name": "Housing", "value": "5000",
    })
    hr_client.post("/hr/add_deduction", data={
        "name": "PAYE", "value": "10", "is_percentage": "y",
    })
    assert AuditLog.query.filter_by(action="Allowance Added").count() == 1
    assert AuditLog.query.filter_by(action="Deduction Added").count() == 1


def test_dashboard_recent_changes_reflects_real_audit_log(hr_client, app):
    """
    Regression: index() used two hardcoded fake entries ("Employee E0001
    marked as inactive...") that never changed no matter what HR actually
    did.
    """
    hr_client.post("/hr/new_employee", data={
        "name": "Dashboard Proof", "role": "nursing", "department": "nursing",
        "job_group": "Group A",
    })
    response = hr_client.get("/hr/")
    assert response.status_code == 200
    assert b"Dashboard Proof" in response.data
    assert b"Employee E0001 marked as inactive" not in response.data


# ── employee_list: dashboard status filters were previously ignored ───────

def test_employee_list_status_filter_matches_dashboard_links(hr_client, app):
    """
    Regression: hr/index.html links to employee_list?status=active and
    ?status=inactive, but the route ignored request.args entirely and
    always returned every employee regardless of which card was clicked.
    """
    _make_employee(employee_id="E-ACT", name="Active One", is_active=True)
    _make_employee(employee_id="E-INACT", name="Inactive One", is_active=False)

    active_resp = hr_client.get("/hr/employee_list?status=active")
    assert b"Active One" in active_resp.data
    assert b"Inactive One" not in active_resp.data

    inactive_resp = hr_client.get("/hr/employee_list?status=inactive")
    assert b"Inactive One" in inactive_resp.data
    assert b"Active One" not in inactive_resp.data


def test_employee_list_search_by_name_or_employee_id(hr_client, app):
    _make_employee(employee_id="E-FIND", name="Findable Employee")
    _make_employee(employee_id="E-OTHER", name="Someone Else")

    by_name = hr_client.get("/hr/employee_list?search=Findable")
    assert b"Findable Employee" in by_name.data
    assert b"Someone Else" not in by_name.data

    by_id = hr_client.get("/hr/employee_list?search=E-FIND")
    assert b"Findable Employee" in by_id.data


def test_employee_list_department_filter(hr_client, app):
    _make_employee(employee_id="E-NUR", name="Nurse Person", department="nursing")
    _make_employee(employee_id="E-LAB", name="Lab Person", department="laboratory")

    response = hr_client.get("/hr/employee_list?department=laboratory")
    assert b"Lab Person" in response.data
    assert b"Nurse Person" not in response.data


# ── Leave balance tracking ─────────────────────────────────────────────────

def test_leave_days_remaining_defaults_to_full_entitlement(app):
    employee = _make_employee(employee_id="E-BAL1", name="Fresh Balance")
    assert employee.annual_leave_days == 21
    assert employee.leave_days_remaining() == 21


def test_leave_days_remaining_only_counts_approved_vacation(app):
    employee = _make_employee(employee_id="E-BAL2", name="Balance Target")
    db.session.add(Leave(employee_id=employee.id,
                         start_date=datetime(2026, 1, 1), end_date=datetime(2026, 1, 5),
                         type="vacation", status="Approved"))  # 5 days
    db.session.add(Leave(employee_id=employee.id,
                         start_date=datetime(2026, 2, 1), end_date=datetime(2026, 2, 2),
                         type="vacation", status="Pending"))  # not counted yet
    db.session.add(Leave(employee_id=employee.id,
                         start_date=datetime(2026, 3, 1), end_date=datetime(2026, 3, 10),
                         type="sick", status="Approved"))  # different type, not counted
    db.session.commit()

    assert employee.leave_days_used() == 5
    assert employee.leave_days_remaining() == 16


def test_leave_request_rejects_end_date_before_start_date(client, app, linked_pair):
    employee, user = linked_pair
    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"})
    response = client.post("/hr/leave_request", data={
        "start_date": "2026-10-10", "end_date": "2026-10-05", "type": "vacation",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert Leave.query.count() == 0


def test_leave_request_rejects_overlapping_pending_request(client, app, linked_pair):
    employee, user = linked_pair
    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"})
    client.post("/hr/leave_request", data={
        "start_date": "2026-10-01", "end_date": "2026-10-10", "type": "vacation",
    })
    assert Leave.query.count() == 1

    response = client.post("/hr/leave_request", data={
        "start_date": "2026-10-05", "end_date": "2026-10-06", "type": "vacation",
    }, follow_redirects=True)
    assert response.status_code == 200
    # Second, overlapping request must not have been saved.
    assert Leave.query.count() == 1


def test_leave_request_rejects_requests_beyond_annual_balance(client, app, linked_pair):
    employee, user = linked_pair
    employee.annual_leave_days = 5
    db.session.commit()

    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"})
    response = client.post("/hr/leave_request", data={
        # 10 inclusive days requested against a 5-day entitlement.
        "start_date": "2026-10-01", "end_date": "2026-10-10", "type": "vacation",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert Leave.query.count() == 0


def test_leave_request_sick_leave_is_not_capped_by_vacation_balance(client, app, linked_pair):
    employee, user = linked_pair
    employee.annual_leave_days = 1
    db.session.commit()

    client.post("/login", data={"username": "linked_nurse", "password": "Nurse!2345"})
    response = client.post("/hr/leave_request", data={
        "start_date": "2026-10-01", "end_date": "2026-10-10", "type": "sick",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert Leave.query.filter_by(type="sick").count() == 1


# ── Payroll idempotency ─────────────────────────────────────────────────────

def test_generate_payroll_twice_updates_instead_of_duplicating(hr_client, app):
    """
    Regression: re-running generate_payroll for a month it had already run
    for inserted a second Payroll row per employee instead of recalculating
    the existing one, silently doubling every reported figure.
    """
    _make_employee(employee_id="E-IDEMP", name="Idempotent Target", basic_salary=10000)

    hr_client.get("/hr/generate_payroll/2026-09")
    assert Payroll.query.filter_by(month="2026-09").count() == 1

    db.session.add(Allowance(job_group="Group A", name="Housing", value=1000))
    db.session.commit()

    hr_client.get("/hr/generate_payroll/2026-09")
    assert Payroll.query.filter_by(month="2026-09").count() == 1  # still one row
    payroll = Payroll.query.filter_by(month="2026-09").first()
    assert float(payroll.gross_pay) == 11000.0  # recalculated with the new allowance
