import json
import logging

from flask import flash, jsonify, render_template, request, session
from flask_login import login_required

from departments.rbac import roles_required

from . import bp  # Import the blueprint

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@bp.route("/ai_discovery", methods=["GET", "POST"])
@login_required
@roles_required("pharmacy", "admin")
def ai_discovery():
    """Renders the AI Drug Discovery page and handles form submissions for candidate generation and docking estimation."""
    from departments.models.compliance import has_ai_consent
    from departments.nlp.src.nvidia_client import NvidiaNIMClient
    from departments.pharmacy.cheminformatics import (
        compute_molecular_properties,
        find_similar_drugs,
        generate_3d_molblock,
    )

    results = None
    tool_used = None

    if request.method == "POST":
        patient_id = (
            request.form.get("patient_id") or request.args.get("patient_id") or ""
        ).strip()
        if patient_id and not has_ai_consent(patient_id):
            return jsonify(
                {
                    "error": "AI-assisted drug discovery unavailable: patient has not consented to AI processing.",
                    "code": "AI_CONSENT_REQUIRED",
                }
            ), 403

        client = NvidiaNIMClient()
        action = request.form.get("action")
        is_offline = not client.is_available()

        if action == "molmim":
            properties = request.form.get("target_properties", "").strip()
            if properties:
                candidates = client.generate_molecules(properties)
                enriched_molecules = []
                discarded_count = 0
                for smiles in candidates:
                    props = compute_molecular_properties(smiles)
                    if props and props.get("is_valid"):
                        # Avoid duplicates in same generation batch
                        if not any(
                            m["smiles"] == props["smiles"] for m in enriched_molecules
                        ):
                            props["molblock_3d"] = generate_3d_molblock(props["smiles"])
                            props["similar_drugs"] = find_similar_drugs(
                                props["smiles"], top_n=3
                            )
                            enriched_molecules.append(props)
                    else:
                        discarded_count += 1

                # If all LLM candidates failed RDKit, use fallback
                if not enriched_molecules:
                    fallback_smiles = [
                        "CC(=O)OC1=CC=CC=C1C(=O)O",
                        "CC(=O)NC1=CC=C(O)C=C1",
                        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                    ]
                    for s in fallback_smiles:
                        p = compute_molecular_properties(s)
                        if p and p.get("is_valid"):
                            p["molblock_3d"] = generate_3d_molblock(p["smiles"])
                            p["similar_drugs"] = find_similar_drugs(
                                p["smiles"], top_n=3
                            )
                            enriched_molecules.append(p)

                total_generated = len(enriched_molecules) + discarded_count
                results = {
                    "molecules": enriched_molecules,
                    "properties": properties,
                    "discarded_count": discarded_count,
                    "total_generated": total_generated,
                    "is_offline": is_offline,
                }
                tool_used = "molmim"
            else:
                flash("Please enter target properties.", "error")

        elif action == "diffdock":
            ligand = request.form.get("ligand_smiles", "").strip()
            protein = request.form.get("protein_sequence", "").strip()
            if ligand and protein:
                ligand_props = compute_molecular_properties(ligand)
                if not ligand_props or not ligand_props.get("is_valid"):
                    flash(
                        f"Invalid ligand SMILES: '{ligand}' could not be parsed by RDKit.",
                        "error",
                    )
                else:
                    ligand_props["molblock_3d"] = generate_3d_molblock(
                        ligand_props["smiles"]
                    )
                    docking_results = client.predict_docking(ligand, protein)
                    similar_drugs = find_similar_drugs(ligand_props["smiles"], top_n=3)
                    results = {
                        "docking": docking_results,
                        "ligand": ligand,
                        "ligand_props": ligand_props,
                        "similar_drugs": similar_drugs,
                        "protein_length": len(protein),
                        "is_offline": is_offline,
                    }
                    tool_used = "diffdock"
            else:
                flash(
                    "Please provide both a ligand SMILES string and a protein sequence.",
                    "error",
                )

    shortlist = session.get("discovery_shortlist", [])
    return render_template(
        "pharmacy/ai_discovery.html",
        results=results,
        tool_used=tool_used,
        shortlist=shortlist,
    )


@bp.route("/ai_discovery/shortlist/add", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def ai_discovery_shortlist_add():
    """Adds a candidate molecule object to the Flask server-side session shortlist."""
    data = request.get_json(silent=True) or request.form.to_dict()
    candidate = data.get("candidate")
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except Exception:
            candidate = None

    if not candidate and isinstance(data, dict) and "smiles" in data:
        candidate = data

    if candidate and isinstance(candidate, dict) and candidate.get("smiles"):
        shortlist = session.get("discovery_shortlist", [])
        if not any(item.get("smiles") == candidate["smiles"] for item in shortlist):
            shortlist.append(candidate)
            session["discovery_shortlist"] = shortlist
            session.modified = True
        return jsonify(
            {"success": True, "count": len(shortlist), "shortlist": shortlist}
        )

    return jsonify({"error": "Invalid candidate data"}), 400


@bp.route("/ai_discovery/shortlist/remove", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def ai_discovery_shortlist_remove():
    """Removes a candidate molecule from the Flask server-side session shortlist."""
    data = request.get_json(silent=True) or request.form.to_dict()
    smiles = data.get("smiles")
    index = data.get("index")
    shortlist = session.get("discovery_shortlist", [])

    if index is not None and str(index).isdigit() and 0 <= int(index) < len(shortlist):
        shortlist.pop(int(index))
        session["discovery_shortlist"] = shortlist
        session.modified = True
    elif smiles:
        shortlist = [item for item in shortlist if item.get("smiles") != smiles]
        session["discovery_shortlist"] = shortlist
        session.modified = True

    return jsonify({"success": True, "count": len(shortlist), "shortlist": shortlist})


@bp.route("/ai_discovery/shortlist/clear", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def ai_discovery_shortlist_clear():
    """Clears the Flask server-side session shortlist."""
    session["discovery_shortlist"] = []
    session.modified = True
    return jsonify({"success": True, "count": 0, "shortlist": []})
