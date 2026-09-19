from flask import jsonify, render_template, request, url_for
from flask_login import login_required
from psycopg2.extras import RealDictCursor

from departments.medicine.orders import fetch_drugs_data
from departments.models.medicine import Medicine, OncologyDrug
from departments.models.pharmacy import Drug as PharmDrug
from departments.nlp.logging_setup import get_logger
from departments.rbac import roles_required
from departments.shared.drugcentral import (
    get_drugcentral_connection as get_db_connection,
)

from . import bp
from .oncology import CLINICAL_READ_ROLES

logger = get_logger(__name__)


@bp.route("/drugs-ref/search", methods=["GET"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def drugs_ref():
    """Drugs reference route with search functionality."""
    search_query = request.args.get("search", "").strip()
    category = request.args.get("category", "").strip() or None
    drugs_data = fetch_drugs_data(search_query, category=category)

    return render_template(
        "medicine/drugs_ref.html",
        drugs_data=drugs_data,
        search_query=search_query,
        selected_category=category,
    )


@bp.route("/drugs-ref/api/autocomplete", methods=["GET"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def drugs_ref_autocomplete():
    """API endpoint for live search autocomplete suggestions."""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2:
        return jsonify([])

    results = fetch_drugs_data(q)
    formatted = []
    seen = set()
    for item in results[:10]:
        name = item.get("generic_name") or item.get("product_name") or ""
        if name and name.upper() not in seen:
            seen.add(name.upper())
            formatted.append({
                "generic_name": name,
                "product_name": item.get("product_name") or "",
                "form": item.get("form") or "",
                "route": item.get("route") or "",
                "url": url_for("medicine.drug_details", drug=name),
            })

    return jsonify(formatted)


def _openfda_drug_lookup(drug_query: str) -> dict | None:
    """
    Query the OpenFDA drug label API as a fallback when DrugCentral is unreachable.

    Returns a dict compatible with the drug_details.html template, or None if
    the drug is not found.

    API docs: https://open.fda.gov/apis/drug/label/
    No API key required for <240 requests/minute.
    """
    import requests as _requests

    # Common pharmaceutical salt suffixes to strip for better API matching
    _SALT_SUFFIXES = (
        "HYDROCHLORIDE", "HCL", "SULFATE", "SULPHATE", "SODIUM", "POTASSIUM",
        "CALCIUM", "MALEATE", "MESYLATE", "FUMARATE", "TARTRATE", "ACETATE",
        "PHOSPHATE", "CITRATE", "BROMIDE", "CHLORIDE", "NITRATE", "SUCCINATE",
        "BESYLATE", "TOSYLATE", "LACTATE", "GLUCONATE", "PAMOATE",
    )

    clean = drug_query.strip()
    # Try with full name first, then stripped base name
    search_terms = [clean]
    upper = clean.upper()
    for suffix in _SALT_SUFFIXES:
        if upper.endswith(" " + suffix):
            base = clean[: -(len(suffix) + 1)].strip()
            if base:
                search_terms.append(base)
            break

    for term in search_terms:
        try:
            url = "https://api.fda.gov/drug/label.json"
            resp = _requests.get(
                url,
                params={
                    "search": f'openfda.generic_name:"{term}"',
                    "limit": 1,
                },
                timeout=5,
            )
            if resp.status_code != 200:
                # Also try by brand_name
                resp = _requests.get(
                    url,
                    params={
                        "search": f'openfda.brand_name:"{term}"',
                        "limit": 1,
                    },
                    timeout=5,
                )
            if resp.status_code != 200:
                continue

            data = resp.json()
            results = data.get("results", [])
            if not results:
                continue

            label = results[0]
            openfda = label.get("openfda", {})

            # Drug name — prefer generic_name from openfda metadata
            generic_names = openfda.get("generic_name", [])
            brand_names = openfda.get("brand_name", [])
            drug_name = (
                generic_names[0] if generic_names
                else (brand_names[0] if brand_names else drug_query)
            )

            # Build ingredients list
            ingredients = []
            for gn in generic_names:
                ingredients.append({
                    "substance_name": gn,
                    "quantity": None,
                    "unit": None,
                })

            # Pharmacological classes
            pharma_classes = []
            for pc in openfda.get("pharm_class_epc", []):
                pharma_classes.append({"class_code": "", "source": "FDA EPC", "name": pc})
            for pc in openfda.get("pharm_class_moa", []):
                pharma_classes.append({"class_code": "", "source": "MoA", "name": pc})
            for pc in openfda.get("pharm_class_pe", []):
                pharma_classes.append({"class_code": "", "source": "PE", "name": pc})

            # Products
            products = []
            for bn in brand_names:
                products.append({
                    "product_name": bn,
                    "generic_name": generic_names[0] if generic_names else bn,
                    "route": ", ".join(openfda.get("route", [])),
                    "form": ", ".join(openfda.get("dosage_form", [])),
                    "strength": None,
                })

            # Approvals
            approvals = []
            for appl_no in openfda.get("application_number", []):
                approvals.append({
                    "approval": appl_no,
                    "applicant": openfda.get("manufacturer_name", [""])[0] if openfda.get("manufacturer_name") else "",
                    "type": "FDA",
                    "orphan": None,
                })

            # Struct row — extract MoA from label text
            moa_text = ""
            if label.get("mechanism_of_action"):
                moa_text = label["mechanism_of_action"][0][:2000]
            elif label.get("clinical_pharmacology"):
                moa_text = label["clinical_pharmacology"][0][:2000]

            struct_row = {
                "name": drug_name,
                "mrdef": moa_text or "Refer to clinical reference manual.",
                "smiles": None,
                "inchi": None,
            }

            return {
                "drug_name": drug_name,
                "struct_row": struct_row,
                "ingredients": ingredients,
                "pharma_classes": pharma_classes,
                "products": products,
                "approvals": approvals,
            }
        except _requests.RequestException:
            continue
        except (KeyError, IndexError, ValueError):
            continue

    return None


@bp.route("/drugs-ref/details/<drug>", methods=["GET"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def drug_details(drug: str):
    """Fetch and display detailed information about a specific active ingredient."""
    normalized_drug = drug.strip().upper()

    # Common pharmaceutical salt suffixes to strip for base-name matching
    _SALT_SUFFIXES = (
        "HYDROCHLORIDE", "HCL", "SULFATE", "SULPHATE", "SODIUM", "POTASSIUM",
        "CALCIUM", "MALEATE", "MESYLATE", "MESILATE", "FUMARATE", "TARTRATE",
        "ACETATE", "PHOSPHATE", "CITRATE", "BROMIDE", "CHLORIDE", "NITRATE",
        "SUCCINATE", "BESYLATE", "BESILATE", "TOSYLATE", "LACTATE", "GLUCONATE",
        "BITARTRATE", "DIHYDROCHLORIDE", "MONOHYDRATE", "DIHYDRATE",
        "TRIHYDRATE", "HEMIHYDRATE", "DECANOATE", "ENANTHATE", "VALERATE",
        "PROPIONATE", "BENZOATE", "OXALATE", "MALONATE", "STEARATE",
        "PALMITATE", "LAURATE", "OLEATE", "PAMOATE", "EMBONATE",
    )

    def _strip_salt(name: str) -> str:
        """Strip trailing pharmaceutical salt suffix from a drug name."""
        upper = name.strip().upper()
        for suffix in _SALT_SUFFIXES:
            if upper.endswith(" " + suffix):
                return upper[: -(len(suffix) + 1)].strip()
        return upper

    try:
        with get_db_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            # Tier 1: Exact match on active_ingredient.substance_name
            cur.execute(
                """
                SELECT DISTINCT struct_id
                FROM active_ingredient
                WHERE UPPER(substance_name) = %s
                LIMIT 1
            """,
                [normalized_drug],
            )
            result = cur.fetchone()

            # Tier 2: Search product table for generic_name or product_name match
            if not result:
                cur.execute(
                    """
                    SELECT DISTINCT s.struct_id
                    FROM product p
                    JOIN struct2obprod s ON s.prod_id = p.id
                    WHERE UPPER(p.generic_name) = %s OR UPPER(p.product_name) = %s
                    LIMIT 1
                """,
                    [normalized_drug, normalized_drug],
                )
                result = cur.fetchone()

            # Tier 2.5: Search structures.name — salt forms like
            # "methylphenidate hydrochloride" are stored here rather than
            # in active_ingredient.substance_name.
            if not result:
                cur.execute(
                    """
                    SELECT DISTINCT id AS struct_id
                    FROM structures
                    WHERE UPPER(name) = %s
                    LIMIT 1
                """,
                    [normalized_drug],
                )
                result = cur.fetchone()

            # Tier 3: Search active_ingredient with ILIKE or partial match
            if not result:
                cur.execute(
                    """
                    SELECT DISTINCT struct_id
                    FROM active_ingredient
                    WHERE UPPER(substance_name) ILIKE %s
                    LIMIT 1
                """,
                    [f"%{normalized_drug}%"],
                )
                result = cur.fetchone()

            # Tier 3.5: Strip pharmaceutical salt suffix and re-search.
            # e.g. "METHYLPHENIDATE HYDROCHLORIDE" → try "METHYLPHENIDATE"
            if not result:
                base_name = _strip_salt(normalized_drug)
                if base_name != normalized_drug:
                    # Try active_ingredient with the base name
                    cur.execute(
                        """
                        SELECT DISTINCT struct_id
                        FROM active_ingredient
                        WHERE UPPER(substance_name) = %s
                           OR UPPER(substance_name) ILIKE %s
                        LIMIT 1
                    """,
                        [base_name, f"%{base_name}%"],
                    )
                    result = cur.fetchone()
                    # Also try structures.name with the base name
                    if not result:
                        cur.execute(
                            """
                            SELECT DISTINCT id AS struct_id
                            FROM structures
                            WHERE UPPER(name) = %s
                               OR UPPER(name) ILIKE %s
                            LIMIT 1
                        """,
                            [base_name, f"%{base_name}%"],
                        )
                        result = cur.fetchone()
                    # Try product table with the base name
                    if not result:
                        cur.execute(
                            """
                            SELECT DISTINCT s.struct_id
                            FROM product p
                            JOIN struct2obprod s ON s.prod_id = p.id
                            WHERE UPPER(p.generic_name) ILIKE %s
                               OR UPPER(p.product_name) ILIKE %s
                            LIMIT 1
                        """,
                            [f"%{base_name}%", f"%{base_name}%"],
                        )
                        result = cur.fetchone()

            # Tier 3.7: Fuzzy partial match on structures.name
            if not result:
                cur.execute(
                    """
                    SELECT DISTINCT id AS struct_id
                    FROM structures
                    WHERE UPPER(name) ILIKE %s
                    LIMIT 1
                """,
                    [f"%{normalized_drug}%"],
                )
                result = cur.fetchone()

            # Tier 4: For compound/multi-ingredient strings (comma or 'and' separated),
            # extract candidate ingredients and find the first matching active ingredient struct_id
            if not result and ("," in drug or " and " in drug.lower() or "/" in drug):
                raw_parts = [
                    p.strip().upper()
                    for item in drug.replace(" and ", ",").split(",")
                    for p in item.split("/")
                    if p.strip()
                ]
                for part in raw_parts:
                    if len(part) < 3:
                        continue
                    cur.execute(
                        """
                        SELECT DISTINCT struct_id
                        FROM active_ingredient
                        WHERE UPPER(substance_name) = %s OR UPPER(substance_name) ILIKE %s
                        LIMIT 1
                    """,
                        [part, f"%{part}%"],
                    )
                    sub_match = cur.fetchone()
                    if sub_match:
                        result = sub_match
                        break

            if result:
                struct_id = result["struct_id"]

                # Structures metadata (Mechanism of Action definition, SMILES, InChI)
                cur.execute(
                    """
                    SELECT id, name, smiles, inchi, mrdef
                    FROM structures
                    WHERE id = %s
                """,
                    [struct_id],
                )
                struct_row = cur.fetchone() or {}

                # Active ingredients
                cur.execute(
                    """
                    SELECT DISTINCT substance_name, quantity, unit
                    FROM active_ingredient
                    WHERE struct_id = %s
                """,
                    [struct_id],
                )
                ingredients = cur.fetchall()

                # ATC Code & Defined Daily Dose (DDD)
                cur.execute(
                    """
                    SELECT atc_code, route, ddd, unit_type
                    FROM atc_ddd
                    WHERE struct_id = %s
                """,
                    [struct_id],
                )
                atc_ddd_rows = cur.fetchall()

                # Pharmacological Classes (FDA EPC, MoA, PE, Mesh)
                cur.execute(
                    """
                    SELECT class_code, source, name
                    FROM pharma_class
                    WHERE struct_id = %s
                """,
                    [struct_id],
                )
                pharma_classes = cur.fetchall()

                # Target Protein Interactions & Action Types
                cur.execute(
                    """
                    SELECT DISTINCT
                        td.name AS target_name,
                        act.organism,
                        at.action_type,
                        act.act_value,
                        act.act_unit
                    FROM act_table_full act
                    JOIN target_dictionary td ON act.target_id = td.id
                    LEFT JOIN action_type at ON act.action_type = at.id::VARCHAR
                    WHERE act.struct_id = %s
                    LIMIT 20
                """,
                    [struct_id],
                )
                targets = cur.fetchall()

                # Adverse Effects (FAERS) split by cohort
                cur.execute(
                    """
                    SELECT meddra_name, drug_ae, llr_threshold, level, 'General' AS cohort
                    FROM faers WHERE struct_id = %s
                    UNION ALL
                    SELECT meddra_name, drug_ae, llr_threshold, level, 'Pediatric' AS cohort
                    FROM faers_ped WHERE struct_id = %s
                    UNION ALL
                    SELECT meddra_name, drug_ae, llr_threshold, level, 'Geriatric' AS cohort
                    FROM faers_ger WHERE struct_id = %s
                    UNION ALL
                    SELECT meddra_name, drug_ae, llr_threshold, level, 'Male' AS cohort
                    FROM faers_male WHERE struct_id = %s
                    UNION ALL
                    SELECT meddra_name, drug_ae, llr_threshold, level, 'Female' AS cohort
                    FROM faers_female WHERE struct_id = %s
                    LIMIT 100
                """,
                    [struct_id] * 5,
                )
                adverse_effects = cur.fetchall()

                # Marketed Products & Formulations
                cur.execute(
                    """
                    SELECT DISTINCT p.product_name, p.generic_name, p.route, p.form, s.strength
                    FROM struct2obprod s
                    JOIN product p ON s.prod_id = p.id
                    WHERE s.struct_id = %s
                    LIMIT 50
                """,
                    [struct_id],
                )
                products = cur.fetchall()

                # FDA Approvals & Patents
                cur.execute(
                    """
                    SELECT approval, applicant, type, orphan
                    FROM approval WHERE struct_id = %s
                """,
                    [struct_id],
                )
                approvals = cur.fetchall()

                cur.execute(
                    """
                    SELECT appl_no, trade_name, patent_no, patent_expire_date
                    FROM ob_patent_view WHERE struct_id = %s
                """,
                    [struct_id],
                )
                patents = cur.fetchall()

                primary_ingredient = (
                    ingredients[0]["substance_name"]
                    if ingredients
                    else (struct_row.get("name") or drug)
                )

                return render_template(
                    "medicine/drug_details.html",
                    drug=primary_ingredient,
                    drug_query=drug,
                    struct_id=struct_id,
                    struct_row=struct_row,
                    ingredients=ingredients,
                    atc_ddd_rows=atc_ddd_rows,
                    pharma_classes=pharma_classes,
                    targets=targets,
                    adverse_effects=adverse_effects,
                    products=products,
                    approvals=approvals,
                    patents=patents,
                    is_local=False,
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "DrugCentral connection unavailable (%s); trying OpenFDA API fallback.",
            exc,
        )

    # ── Tier 5: OpenFDA REST API fallback ──────────────────────────────────
    # When DrugCentral is completely unreachable, query the free FDA drug
    # labeling API for basic drug information before giving up.
    try:
        openfda_data = _openfda_drug_lookup(drug)
        if openfda_data:
            return render_template(
                "medicine/drug_details.html",
                drug=openfda_data["drug_name"],
                drug_query=drug,
                struct_id=None,
                struct_row=openfda_data.get("struct_row", {}),
                ingredients=openfda_data.get("ingredients", []),
                atc_ddd_rows=[],
                pharma_classes=openfda_data.get("pharma_classes", []),
                targets=[],
                adverse_effects=[],
                products=openfda_data.get("products", []),
                approvals=openfda_data.get("approvals", []),
                patents=[],
                is_local=False,
                openfda_source=True,
            )
    except Exception as fda_exc:  # noqa: BLE001
        logger.warning("OpenFDA fallback also failed (%s); using local database.", fda_exc)

    # Local hospital database fallback
    candidates = [drug]
    if "," in drug or " and " in drug.lower() or "/" in drug:
        for item in drug.replace(" and ", ",").split(","):
            for part in item.split("/"):
                cleaned = part.strip()
                if len(cleaned) >= 3:
                    candidates.append(cleaned)

    onco = None
    med = None
    pharm = None

    for cand in candidates:
        onco = OncologyDrug.query.filter(OncologyDrug.name.ilike(cand)).first()
        med = Medicine.query.filter(
            (Medicine.generic_name.ilike(cand)) | (Medicine.brand_name.ilike(cand))
        ).first()
        pharm = PharmDrug.query.filter(PharmDrug.generic_name.ilike(cand)).first()
        if onco or med or pharm:
            break

    if onco or med or pharm:
        name = onco.name if onco else (med.generic_name if med else pharm.generic_name)
        form = (
            onco.dosage_form
            if onco
            else (med.dosage if med else pharm.dosage_form)
        )
        strength = onco.strength if onco else (pharm.strength if pharm else "Standard")
        moa = (
            onco.mechanism_of_action
            if (onco and onco.mechanism_of_action)
            else "Refer to clinical reference manual."
        )

        grouped_data = {}
        if onco and onco.side_effects:
            grouped_data["Side Effects"] = [{"description": onco.side_effects}]
        if onco and onco.therapeutic_class:
            grouped_data["Pharmacological Class"] = [
                {"class_name": onco.therapeutic_class, "category": "Oncology"}
            ]

        additional_details = [
            {
                "active_ingredient": name,
                "target_protein": form or "N/A",
                "action_type": strength or "N/A",
                "mechanism_of_action": moa,
            }
        ]
        return render_template(
            "medicine/drug_details.html",
            drug=name,
            additional_details=additional_details,
            grouped_data=grouped_data,
            struct_id=None,
            is_local=True,
        )

    return render_template(
        "medicine/error.html",
        message=f"No details found for {drug}.",
    )
