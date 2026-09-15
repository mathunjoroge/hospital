"""p0: migrate Float financial fields to Numeric and add stock check constraints

Revision ID: f1a2b3c4d5e6
Revises: ebed81b14204
Create Date: 2026-09-15 00:00:00.000000

Changes:
  - drugs.buying_price / selling_price: Float → Numeric(12,2)
  - purchases.unit_cost / total_cost: Float → Numeric(12,2)
  - payrolls.gross_pay / total_deductions / net_pay: Float → Numeric(12,2)
  - allowances.value: Float → Numeric(12,2)
  - deductions.value: Float → Numeric(12,4)  (rates like PAYE need 4dp)
  - custom_rules.value: Float → Numeric(12,4)
  - labtests.cost: Float → Numeric(10,2)
  - imaging.cost: Float → Numeric(10,2)
  - clinics.fee: Float → Numeric(10,2)
  - drugs.quantity_in_stock >= 0  (CheckConstraint)
  - drugs.buying_price >= 0       (CheckConstraint)
  - drugs.selling_price >= 0      (CheckConstraint)
  - batches.quantity_in_stock >= 0 (CheckConstraint)
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f1a2b3c4d5e6"
down_revision = "e789f0123456"
branch_labels = None
depends_on = None


def upgrade():
    # ── drugs ──────────────────────────────────────────────────────────────
    with op.batch_alter_table("drugs") as batch_op:
        batch_op.alter_column(
            "buying_price",
            type_=sa.Numeric(12, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "selling_price",
            type_=sa.Numeric(12, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )
        batch_op.create_check_constraint(
            "ck_drug_stock_non_negative", "quantity_in_stock >= 0"
        )
        batch_op.create_check_constraint(
            "ck_drug_buying_price_non_negative", "buying_price >= 0"
        )
        batch_op.create_check_constraint(
            "ck_drug_selling_price_non_negative", "selling_price >= 0"
        )

    # ── batches ────────────────────────────────────────────────────────────
    with op.batch_alter_table("batches") as batch_op:
        batch_op.create_check_constraint(
            "ck_batch_stock_non_negative", "quantity_in_stock >= 0"
        )

    # ── purchases ──────────────────────────────────────────────────────────
    with op.batch_alter_table("purchases") as batch_op:
        batch_op.alter_column(
            "unit_cost",
            type_=sa.Numeric(12, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "total_cost",
            type_=sa.Numeric(12, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )

    # ── allowances ─────────────────────────────────────────────────────────
    with op.batch_alter_table("allowances") as batch_op:
        batch_op.alter_column(
            "value",
            type_=sa.Numeric(12, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )

    # ── payrolls ───────────────────────────────────────────────────────────
    with op.batch_alter_table("payrolls") as batch_op:
        batch_op.alter_column(
            "gross_pay",
            type_=sa.Numeric(12, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "total_deductions",
            type_=sa.Numeric(12, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "net_pay",
            type_=sa.Numeric(12, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )

    # ── deductions ─────────────────────────────────────────────────────────
    with op.batch_alter_table("deductions") as batch_op:
        batch_op.alter_column(
            "value",
            type_=sa.Numeric(12, 4),
            existing_type=sa.Float(),
            existing_nullable=False,
        )

    # ── custom_rules ───────────────────────────────────────────────────────
    with op.batch_alter_table("custom_rules") as batch_op:
        batch_op.alter_column(
            "value",
            type_=sa.Numeric(12, 4),
            existing_type=sa.Float(),
            existing_nullable=False,
        )

    # ── labtests ───────────────────────────────────────────────────────────
    with op.batch_alter_table("labtests") as batch_op:
        batch_op.alter_column(
            "cost",
            type_=sa.Numeric(10, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )

    # ── imaging ────────────────────────────────────────────────────────────
    with op.batch_alter_table("imaging") as batch_op:
        batch_op.alter_column(
            "cost",
            type_=sa.Numeric(10, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )

    # ── clinics ────────────────────────────────────────────────────────────
    with op.batch_alter_table("clinics") as batch_op:
        batch_op.alter_column(
            "fee",
            type_=sa.Numeric(10, 2),
            existing_type=sa.Float(),
            existing_nullable=False,
        )


def downgrade():
    # Revert Numeric → Float (precision/constraint info is lost on downgrade)

    with op.batch_alter_table("clinics") as batch_op:
        batch_op.alter_column("fee", type_=sa.Float(), existing_nullable=False)

    with op.batch_alter_table("imaging") as batch_op:
        batch_op.alter_column("cost", type_=sa.Float(), existing_nullable=False)

    with op.batch_alter_table("labtests") as batch_op:
        batch_op.alter_column("cost", type_=sa.Float(), existing_nullable=False)

    with op.batch_alter_table("custom_rules") as batch_op:
        batch_op.alter_column("value", type_=sa.Float(), existing_nullable=False)

    with op.batch_alter_table("deductions") as batch_op:
        batch_op.alter_column("value", type_=sa.Float(), existing_nullable=False)

    with op.batch_alter_table("payrolls") as batch_op:
        batch_op.alter_column("gross_pay", type_=sa.Float(), existing_nullable=False)
        batch_op.alter_column("total_deductions", type_=sa.Float(), existing_nullable=False)
        batch_op.alter_column("net_pay", type_=sa.Float(), existing_nullable=False)

    with op.batch_alter_table("allowances") as batch_op:
        batch_op.alter_column("value", type_=sa.Float(), existing_nullable=False)

    with op.batch_alter_table("purchases") as batch_op:
        batch_op.alter_column("unit_cost", type_=sa.Float(), existing_nullable=False)
        batch_op.alter_column("total_cost", type_=sa.Float(), existing_nullable=False)

    with op.batch_alter_table("batches") as batch_op:
        batch_op.drop_constraint("ck_batch_stock_non_negative", type_="check")

    with op.batch_alter_table("drugs") as batch_op:
        batch_op.drop_constraint("ck_drug_selling_price_non_negative", type_="check")
        batch_op.drop_constraint("ck_drug_buying_price_non_negative", type_="check")
        batch_op.drop_constraint("ck_drug_stock_non_negative", type_="check")
        batch_op.alter_column("selling_price", type_=sa.Float(), existing_nullable=False)
        batch_op.alter_column("buying_price", type_=sa.Float(), existing_nullable=False)
