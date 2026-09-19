"""oncology module hardening: chemo orders table, note void columns, stage width

1. chemotherapy_regimen_orders
   The model existed but NO migration ever created the table, so any database
   built with `flask db upgrade` had no chemotherapy order table (the cumulative
   lifetime-dose check then silently returned "no history").  Created here if
   missing; if it already exists (databases bootstrapped with db.create_all())
   the new columns are added instead.  Also adds a partial unique index that
   allows only one *active* (non-cancelled) order per patient/protocol/cycle.
2. oncology_notes: is_voided / voided_by / voided_at / voided_reason.
   The void route assigned these attributes but they were never mapped columns,
   so "voiding" a note persisted nothing.
3. onco_patients.stage widened 20 -> 100 (CancerStage.label is 100 chars).

Downgrade removes only the columns/index this revision added and deliberately
does NOT drop chemotherapy_regimen_orders (clinical order history), its patient
foreign key (a harmless integrity constraint), or narrow onco_patients.stage
(possible truncation).

Revision ID: e5f6a7b8c9d0
Revises: c3d4e5f6a7b8
Create Date: 2026-09-19

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "e5f6a7b8c9d0"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None

CHEMO = "chemotherapy_regimen_orders"
ACTIVE_CYCLE_INDEX = "uq_chemo_active_patient_protocol_cycle"


def _columns(inspector, table):
    return {c["name"]: c for c in inspector.get_columns(table)}


def _fk_name(inspector, table, column):
    """Actual name of the FK on `column` (None if absent or unnamed, e.g. SQLite)."""
    for fk in inspector.get_foreign_keys(table):
        if fk.get("constrained_columns") == [column]:
            return fk.get("name")
    return None


def _create_chemo_table():
    op.create_table(
        CHEMO,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.String(length=20), nullable=False),
        sa.Column("physician_id", sa.Integer(), nullable=False),
        sa.Column("protocol_name", sa.String(length=50), nullable=False),
        sa.Column("cancer_type", sa.String(length=100), nullable=True),
        sa.Column("weight_kg", sa.Float(), nullable=False),
        sa.Column("height_cm", sa.Float(), nullable=False),
        sa.Column("bsa_m2", sa.Float(), nullable=False),
        sa.Column("bsa_formula", sa.String(length=20), nullable=False),
        sa.Column("cycle_number", sa.Integer(), nullable=False),
        sa.Column("total_cycles", sa.Integer(), nullable=False),
        sa.Column("calculated_doses_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("has_toxicity_warning", sa.Boolean(), nullable=False),
        sa.Column("toxicity_warning_details", sa.Text(), nullable=True),
        sa.Column("toxicity_override_reason", sa.Text(), nullable=True),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column("status_changed_by", sa.Integer(), nullable=True),
        sa.Column("status_changed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.patient_id"]),
        sa.ForeignKeyConstraint(["physician_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["status_changed_by"],
            ["users.id"],
            name="fk_chemo_orders_status_changed_by_users",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chemotherapy_regimen_orders_patient_id"), CHEMO, ["patient_id"]
    )
    op.create_index(
        op.f("ix_chemotherapy_regimen_orders_created_at"), CHEMO, ["created_at"]
    )


def _ensure_chemo_patient_fk(bind, inspector):
    """
    Pre-existing table (db.create_all() era) has no patient FK. Add it so that
    NEW orders can never reference a non-existent patient. On PostgreSQL it is
    added NOT VALID: enforced for every new/updated row, while any legacy
    orphaned rows are tolerated (and can be reviewed, then VALIDATEd, by an
    operator) instead of blocking the upgrade.
    """
    if _fk_name(inspector, CHEMO, "patient_id"):
        return
    name = "fk_chemo_orders_patient_id_patients"
    if bind.dialect.name == "postgresql":
        op.create_foreign_key(
            name,
            CHEMO,
            "patients",
            ["patient_id"],
            ["patient_id"],
            postgresql_not_valid=True,
        )
    else:
        with op.batch_alter_table(CHEMO, schema=None) as batch_op:
            batch_op.create_foreign_key(
                name, "patients", ["patient_id"], ["patient_id"]
            )


def _add_missing_chemo_columns(inspector):
    existing = _columns(inspector, CHEMO)
    wanted = [
        sa.Column("toxicity_override_reason", sa.Text(), nullable=True),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column("status_changed_by", sa.Integer(), nullable=True),
        sa.Column("status_changed_at", sa.DateTime(), nullable=True),
    ]
    missing = [c for c in wanted if c.name not in existing]
    if not missing:
        return
    with op.batch_alter_table(CHEMO, schema=None) as batch_op:
        for col in missing:
            batch_op.add_column(col)
        if any(c.name == "status_changed_by" for c in missing):
            batch_op.create_foreign_key(
                "fk_chemo_orders_status_changed_by_users",
                "users",
                ["status_changed_by"],
                ["id"],
            )


def _create_active_cycle_index(bind, inspector):
    if ACTIVE_CYCLE_INDEX in {i["name"] for i in inspector.get_indexes(CHEMO)}:
        return
    duplicates = bind.execute(
        sa.text(
            "SELECT patient_id, protocol_name, cycle_number, COUNT(*) AS n "
            f"FROM {CHEMO} WHERE status <> 'CANCELLED' "
            "GROUP BY patient_id, protocol_name, cycle_number HAVING COUNT(*) > 1"
        )
    ).fetchall()
    if duplicates:
        # Never silently cancel/alter clinical orders in a migration: a human
        # must decide which duplicate is the real one.
        listing = "; ".join(f"{r[0]}/{r[1]}/cycle {r[2]} x{r[3]}" for r in duplicates)
        raise RuntimeError(
            "Cannot enforce one active chemotherapy order per patient/protocol/cycle: "
            f"duplicate active orders exist ({listing}). Cancel the erroneous "
            "duplicates (status='CANCELLED', with a status_reason) and re-run the upgrade."
        )
    op.create_index(
        ACTIVE_CYCLE_INDEX,
        CHEMO,
        ["patient_id", "protocol_name", "cycle_number"],
        unique=True,
        postgresql_where=sa.text("status <> 'CANCELLED'"),
        sqlite_where=sa.text("status <> 'CANCELLED'"),
    )


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # 1. chemotherapy_regimen_orders --------------------------------------
    if CHEMO not in tables:
        _create_chemo_table()
    else:
        _add_missing_chemo_columns(inspector)
        _ensure_chemo_patient_fk(bind, sa.inspect(bind))
    _create_active_cycle_index(bind, sa.inspect(bind))

    # 2. oncology_notes soft-void columns ---------------------------------
    if "oncology_notes" in tables:
        existing = _columns(inspector, "oncology_notes")
        with op.batch_alter_table("oncology_notes", schema=None) as batch_op:
            if "is_voided" not in existing:
                batch_op.add_column(
                    sa.Column(
                        "is_voided",
                        sa.Boolean(),
                        nullable=False,
                        server_default=sa.false(),
                    )
                )
            if "voided_by" not in existing:
                batch_op.add_column(sa.Column("voided_by", sa.Integer(), nullable=True))
                batch_op.create_foreign_key(
                    "fk_oncology_notes_voided_by_users", "users", ["voided_by"], ["id"]
                )
            if "voided_at" not in existing:
                batch_op.add_column(
                    sa.Column("voided_at", sa.DateTime(), nullable=True)
                )
            if "voided_reason" not in existing:
                batch_op.add_column(
                    sa.Column("voided_reason", sa.String(length=500), nullable=True)
                )

    # 3. onco_patients.stage width ----------------------------------------
    if "onco_patients" in tables:
        stage = _columns(inspector, "onco_patients").get("stage")
        current_len = getattr(stage["type"], "length", None) if stage else None
        if stage is not None and current_len is not None and current_len < 100:
            with op.batch_alter_table("onco_patients", schema=None) as batch_op:
                batch_op.alter_column(
                    "stage",
                    existing_type=sa.String(length=current_len),
                    type_=sa.String(length=100),
                    existing_nullable=False,
                )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "oncology_notes" in tables:
        existing = _columns(inspector, "oncology_notes")
        with op.batch_alter_table("oncology_notes", schema=None) as batch_op:
            fk = _fk_name(inspector, "oncology_notes", "voided_by")
            if fk:
                batch_op.drop_constraint(fk, type_="foreignkey")
            for name in ("voided_reason", "voided_at", "voided_by", "is_voided"):
                if name in existing:
                    batch_op.drop_column(name)

    if CHEMO in tables:
        if ACTIVE_CYCLE_INDEX in {i["name"] for i in inspector.get_indexes(CHEMO)}:
            op.drop_index(ACTIVE_CYCLE_INDEX, table_name=CHEMO)
        existing = _columns(inspector, CHEMO)
        with op.batch_alter_table(CHEMO, schema=None) as batch_op:
            fk = _fk_name(inspector, CHEMO, "status_changed_by")
            if fk:
                batch_op.drop_constraint(fk, type_="foreignkey")
            for name in (
                "status_changed_at",
                "status_changed_by",
                "status_reason",
                "toxicity_override_reason",
            ):
                if name in existing:
                    batch_op.drop_column(name)
    # NOTE: chemotherapy_regimen_orders itself and onco_patients.stage width are
    # intentionally left in place (clinical data / possible truncation).
