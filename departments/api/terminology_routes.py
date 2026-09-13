"""
departments/api/terminology_routes.py
──────────────────────────────────────
API endpoints for terminology lookup and autocomplete search.
"""

from flask import jsonify, request

from departments.medicine.terminology_server import FHIRTerminologyServer
from extensions import limiter

from . import bp


@bp.route("/terminology/search", methods=["GET"])
@limiter.limit("60 per minute")
def api_terminology_search():
    """
    Fast clinical dropdown autocomplete endpoint across ICD-10, SNOMED CT, and LOINC.

    Query parameters:
      - q / query: search term (code or text description)
      - system: 'ICD10', 'SNOMED', 'LOINC', or 'ALL'
      - limit: max number of results (default 20)
    """
    query = request.args.get("q") or request.args.get("query")
    system = request.args.get("system")
    limit_arg = request.args.get("limit", "20")
    try:
        limit = int(limit_arg)
    except ValueError:
        limit = 20

    results = FHIRTerminologyServer.search_terms(query, system, limit=limit)
    return jsonify(results), 200
