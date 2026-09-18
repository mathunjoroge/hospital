"""
tests/test_terminology_importers.py
────────────────────────────────────
Unit tests for the clinical terminology stack:
  - departments/medicine/who_icd_client.py
  - departments/medicine/umls_client.py
  - departments/medicine/icd10_importer.py
  - departments/medicine/snomed_importer.py
  - departments/medicine/loinc_importer.py
  - departments/tasks.py  (sync_icd10_codes, sync_snomed_codes, sync_loinc_codes)

All external HTTP calls are mocked; no real WHO or UMLS API is ever contacted.

Key mock-target notes
─────────────────────
* who_icd_client: module-level globals (_cached_token, _token_expires_at) are
  reset between tests via _reset_who_token_cache() to prevent cross-test
  contamination.

* icd10_importer.import_from_who_api() does a LOCAL import of walk_icd10_tree:
    `from departments.medicine.who_icd_client import walk_icd10_tree`
  Therefore the correct patch target is
    `departments.medicine.who_icd_client.walk_icd10_tree`
  NOT `departments.medicine.icd10_importer.walk_icd10_tree`.

* umls_client.get_umls_api_key() wraps `current_app.config.get(...)` in a
  try/except RuntimeError, so outside a Flask context it naturally falls back
  to os.getenv(). We rely on that native behaviour instead of patching the
  Flask LocalProxy object.

* Celery tasks (sync_*) each do `from app import app` and open their own
  app_context(). In tests we patch the importer functions at their
  *call site* inside the task body rather than trying to swap the Flask app
  object in sys.modules.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests


# ---------------------------------------------------------------------------
# Helper: reset the in-process WHO Bearer-token cache between tests so that
# one test's cached token cannot contaminate the next.
# ---------------------------------------------------------------------------
def _reset_who_token_cache():
    import departments.medicine.who_icd_client as wic

    wic._cached_token = None
    wic._token_expires_at = 0.0


# ===========================================================================
# 1. WHO ICD-10 Client
# ===========================================================================


class TestWhoIcdClient:
    """departments/medicine/who_icd_client.py"""

    def setup_method(self):
        _reset_who_token_cache()

    def teardown_method(self):
        _reset_who_token_cache()

    # -- _force_https --------------------------------------------------------

    def test_force_https_upgrades_http_url(self):
        from departments.medicine.who_icd_client import _force_https

        assert _force_https("http://id.who.int/icd/release/10/2019") == (
            "https://id.who.int/icd/release/10/2019"
        )

    def test_force_https_leaves_https_url_unchanged(self):
        from departments.medicine.who_icd_client import _force_https

        url = "https://id.who.int/icd/release/10/2019"
        assert _force_https(url) == url

    # -- _get_bearer_token ---------------------------------------------------

    def test_get_bearer_token_raises_when_credentials_absent(self, monkeypatch):
        monkeypatch.delenv("WHO_ICD_CLIENT_ID", raising=False)
        monkeypatch.delenv("WHO_ICD_CLIENT_SECRET", raising=False)
        from departments.medicine.who_icd_client import _get_bearer_token

        with pytest.raises(RuntimeError, match="WHO_ICD_CLIENT_ID"):
            _get_bearer_token()

    def test_get_bearer_token_fetches_and_stores_token(self, monkeypatch):
        monkeypatch.setenv("WHO_ICD_CLIENT_ID", "test_id")
        monkeypatch.setenv("WHO_ICD_CLIENT_SECRET", "test_secret")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"access_token": "tok123", "expires_in": 3600}
        mock_resp.raise_for_status.return_value = None

        with patch(
            "departments.medicine.who_icd_client.requests.post",
            return_value=mock_resp,
        ) as mock_post:
            from departments.medicine.who_icd_client import _get_bearer_token

            token = _get_bearer_token()

        assert token == "tok123"
        mock_post.assert_called_once()
        import departments.medicine.who_icd_client as wic

        assert wic._cached_token == "tok123"

    def test_get_bearer_token_reuses_unexpired_cache(self, monkeypatch):
        monkeypatch.setenv("WHO_ICD_CLIENT_ID", "test_id")
        monkeypatch.setenv("WHO_ICD_CLIENT_SECRET", "test_secret")

        import departments.medicine.who_icd_client as wic

        wic._cached_token = "cached_token"
        wic._token_expires_at = time.time() + 3000

        with patch("departments.medicine.who_icd_client.requests.post") as mock_post:
            token = wic._get_bearer_token()

        assert token == "cached_token"
        mock_post.assert_not_called()

    # -- _api_get ------------------------------------------------------------

    def test_api_get_attaches_bearer_header_and_returns_json(self, monkeypatch):
        monkeypatch.setenv("WHO_ICD_CLIENT_ID", "test_id")
        monkeypatch.setenv("WHO_ICD_CLIENT_SECRET", "test_secret")

        token_resp = MagicMock()
        token_resp.json.return_value = {
            "access_token": "bearer_xyz",
            "expires_in": 3600,
        }
        token_resp.raise_for_status.return_value = None

        api_resp = MagicMock()
        api_resp.json.return_value = {"code": "A01", "title": {"@value": "Typhoid"}}
        api_resp.raise_for_status.return_value = None

        with patch(
            "departments.medicine.who_icd_client.requests.post",
            return_value=token_resp,
        ), patch(
            "departments.medicine.who_icd_client.requests.get",
            return_value=api_resp,
        ):
            from departments.medicine.who_icd_client import _api_get

            result = _api_get("https://id.who.int/icd/release/10/2019")

        assert result["code"] == "A01"

    def test_api_get_exhausts_retries_then_raises(self, monkeypatch):
        monkeypatch.setenv("WHO_ICD_CLIENT_ID", "test_id")
        monkeypatch.setenv("WHO_ICD_CLIENT_SECRET", "test_secret")

        token_resp = MagicMock()
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 3600}
        token_resp.raise_for_status.return_value = None

        with patch(
            "departments.medicine.who_icd_client.requests.post",
            return_value=token_resp,
        ), patch(
            "departments.medicine.who_icd_client.requests.get",
            side_effect=requests.RequestException("timeout"),
        ), patch("departments.medicine.who_icd_client.time.sleep"):
            from departments.medicine.who_icd_client import _api_get

            with pytest.raises(RuntimeError, match="WHO API request failed after"):
                _api_get("https://id.who.int/icd/release/10/2019", retries=2)

    def test_api_get_clears_token_cache_on_401(self, monkeypatch):
        monkeypatch.setenv("WHO_ICD_CLIENT_ID", "test_id")
        monkeypatch.setenv("WHO_ICD_CLIENT_SECRET", "test_secret")

        import departments.medicine.who_icd_client as wic

        wic._cached_token = "stale_token"
        wic._token_expires_at = time.time() + 3000

        http_resp_401 = MagicMock()
        http_resp_401.status_code = 401
        http_error = requests.HTTPError(response=http_resp_401)

        token_resp = MagicMock()
        token_resp.json.return_value = {"access_token": "new_tok", "expires_in": 3600}
        token_resp.raise_for_status.return_value = None

        with patch(
            "departments.medicine.who_icd_client.requests.get",
            side_effect=http_error,
        ), patch(
            "departments.medicine.who_icd_client.requests.post",
            return_value=token_resp,
        ), patch("departments.medicine.who_icd_client.time.sleep"), pytest.raises(
            RuntimeError
        ):
            wic._api_get("https://id.who.int/icd/release/10/2019", retries=2)

        # Cache must have been cleared by the 401 handler
        assert wic._cached_token is None

    # -- walk_icd10_tree -----------------------------------------------------

    def test_walk_icd10_tree_yields_block_and_leaf_codes(self, monkeypatch):
        """Minimal tree: root → chapter I → block A00-A09 → leaf A01."""
        monkeypatch.setenv("WHO_ICD_CLIENT_ID", "test_id")
        monkeypatch.setenv("WHO_ICD_CLIENT_SECRET", "test_secret")

        # _api_get responses consumed in call order: root, chapter, block, leaf
        responses = [
            {"child": ["https://id.who.int/icd/release/10/2019/I"]},
            {
                "title": {"@value": "Infectious"},
                "code": "I",
                "child": ["https://id.who.int/icd/release/10/2019/A00-A09"],
            },
            {
                "title": {"@value": "Intestinal infections"},
                "code": "A00-A09",
                "child": ["https://id.who.int/icd/release/10/2019/A01"],
            },
            {"title": {"@value": "Typhoid fever"}, "code": "A01", "child": []},
        ]

        def _fake_api_get(url, params=None, retries=3):
            return responses.pop(0)

        with patch(
            "departments.medicine.who_icd_client._api_get",
            side_effect=_fake_api_get,
        ), patch("departments.medicine.who_icd_client.time.sleep"):
            from departments.medicine.who_icd_client import walk_icd10_tree

            results = list(walk_icd10_tree("2019"))

        codes = [r[0] for r in results]
        # Chapter "I" is a roman numeral → excluded; block and leaf must appear
        assert "A00-A09" in codes
        assert "A01" in codes
        assert "I" not in codes

    def test_walk_icd10_tree_skips_all_22_roman_chapter_codes(self, monkeypatch):
        monkeypatch.setenv("WHO_ICD_CLIENT_ID", "test_id")
        monkeypatch.setenv("WHO_ICD_CLIENT_SECRET", "test_secret")
        roman = [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
            "XII",
            "XIII",
            "XIV",
            "XV",
            "XVI",
            "XVII",
            "XVIII",
            "XIX",
            "XX",
            "XXI",
            "XXII",
        ]
        responses = [{"child": [f"https://x/{c}" for c in roman]}]
        for r in roman:
            responses.append(
                {"title": {"@value": f"Chapter {r}"}, "code": r, "child": []}
            )

        def _fake(url, params=None, retries=3):
            return responses.pop(0)

        with patch(
            "departments.medicine.who_icd_client._api_get", side_effect=_fake
        ), patch("departments.medicine.who_icd_client.time.sleep"):
            from departments.medicine.who_icd_client import walk_icd10_tree

            assert list(walk_icd10_tree("2019")) == []

    # -- search_icd10_live ---------------------------------------------------

    def test_search_icd10_live_returns_normalised_results(self):
        data = {
            "destinationEntities": [
                {"theCode": "A01.0", "title": "Typhoid fever", "chapter": "I"},
                {"theCode": "A01.1", "title": "Paratyphoid fever A", "chapter": "I"},
            ]
        }
        with patch("departments.medicine.who_icd_client._api_get", return_value=data):
            from departments.medicine.who_icd_client import search_icd10_live

            results = search_icd10_live("typhoid")

        assert len(results) == 2
        assert results[0] == {
            "code": "A01.0",
            "description": "Typhoid fever",
            "category": "I",
        }

    def test_search_icd10_live_empty_query_returns_empty(self):
        from departments.medicine.who_icd_client import search_icd10_live

        assert search_icd10_live("") == []

    def test_search_icd10_live_api_error_returns_empty(self):
        with patch(
            "departments.medicine.who_icd_client._api_get",
            side_effect=RuntimeError("connection refused"),
        ):
            from departments.medicine.who_icd_client import search_icd10_live

            assert search_icd10_live("typhoid") == []

    def test_search_icd10_live_title_list_is_joined_with_pipe(self):
        """When `title` is a list its elements are joined with ' | '."""
        data = {
            "destinationEntities": [
                {"theCode": "Z99", "title": ["Primary", "Secondary"], "chapter": "XXII"}
            ]
        }
        with patch("departments.medicine.who_icd_client._api_get", return_value=data):
            from departments.medicine.who_icd_client import search_icd10_live

            results = search_icd10_live("test")
        assert results[0]["description"] == "Primary | Secondary"


# ===========================================================================
# 2. UMLS Client
# ===========================================================================
# NOTE: umls_client.get_umls_api_key() does `current_app.config.get(...)` inside
# a try/except RuntimeError block. Outside a Flask app context, current_app raises
# RuntimeError, which the function catches and falls through to os.getenv().
# We rely on that native behaviour instead of trying to mock the LocalProxy.


class TestUmlsClient:
    """departments/medicine/umls_client.py"""

    # -- get_umls_api_key ----------------------------------------------------

    def test_get_umls_api_key_reads_env_outside_flask(self, monkeypatch):
        # No app fixture → no app context → LocalProxy raises RuntimeError →
        # function falls through to os.getenv()
        monkeypatch.setenv("UMLS_API_KEY", "env_key_123")
        from departments.medicine.umls_client import get_umls_api_key

        assert get_umls_api_key() == "env_key_123"

    def test_get_umls_api_key_reads_flask_config_inside_context(self, app):
        app.config["UMLS_API_KEY"] = "flask_key_456"
        with app.app_context():
            from departments.medicine.umls_client import get_umls_api_key

            assert get_umls_api_key() == "flask_key_456"

    def test_get_umls_api_key_returns_empty_when_missing(self, monkeypatch):
        monkeypatch.delenv("UMLS_API_KEY", raising=False)
        from departments.medicine.umls_client import get_umls_api_key

        assert get_umls_api_key() == ""

    # -- search_umls ---------------------------------------------------------

    def test_search_umls_returns_empty_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("UMLS_API_KEY", raising=False)
        from departments.medicine.umls_client import search_umls

        assert search_umls("fever") == []

    def test_search_umls_returns_empty_for_blank_query(self, monkeypatch):
        monkeypatch.setenv("UMLS_API_KEY", "test_key")
        from departments.medicine.umls_client import search_umls

        assert search_umls("") == []
        assert search_umls("   ") == []

    def test_search_umls_returns_normalised_results(self, monkeypatch):
        monkeypatch.setenv("UMLS_API_KEY", "test_key")

        search_payload = {"result": {"results": [{"name": "Fever", "ui": "C0015967"}]}}
        search_resp = MagicMock()
        search_resp.json.return_value = search_payload
        search_resp.raise_for_status.return_value = None

        atom_resp = MagicMock()
        atom_resp.status_code = 200
        atom_resp.json.return_value = {
            "result": [
                {
                    "rootSource": "SNOMEDCT_US",
                    "code": "https://snomed.info/id/424754009",
                }
            ]
        }

        with patch(
            "departments.medicine.umls_client.requests.get",
            side_effect=[search_resp, atom_resp],
        ):
            from departments.medicine.umls_client import search_umls

            results = search_umls("fever", sab="SNOMEDCT_US")

        assert len(results) == 1
        assert results[0]["code"] == "424754009"
        assert results[0]["description"] == "Fever"

    def test_search_umls_connection_error_returns_empty(self, monkeypatch):
        monkeypatch.setenv("UMLS_API_KEY", "test_key")
        with patch(
            "departments.medicine.umls_client.requests.get",
            side_effect=requests.ConnectionError("refused"),
        ):
            from departments.medicine.umls_client import search_umls

            assert search_umls("fever") == []

    # -- _extract_code_for_cui -----------------------------------------------

    def test_extract_code_for_cui_parses_snomed_url_path(self):
        atom_resp = MagicMock()
        atom_resp.status_code = 200
        atom_resp.json.return_value = {
            "result": [
                {
                    "rootSource": "SNOMEDCT_US",
                    "code": "https://snomed.info/id/424754009",
                }
            ]
        }
        with patch(
            "departments.medicine.umls_client.requests.get",
            return_value=atom_resp,
        ):
            from departments.medicine.umls_client import _extract_code_for_cui

            assert (
                _extract_code_for_cui("C0015967", "SNOMEDCT_US", "key") == "424754009"
            )

    def test_extract_code_for_cui_non_c_id_returns_as_is(self):
        from departments.medicine.umls_client import _extract_code_for_cui

        assert _extract_code_for_cui("424754009", "SNOMEDCT_US", "key") == "424754009"

    def test_extract_code_for_cui_network_failure_returns_none(self):
        with patch(
            "departments.medicine.umls_client.requests.get",
            side_effect=requests.ConnectionError("refused"),
        ):
            from departments.medicine.umls_client import _extract_code_for_cui

            assert _extract_code_for_cui("C0015967", "SNOMEDCT_US", "key") is None

    # -- search_snomed_live / search_loinc_live ------------------------------

    def test_search_snomed_live_routes_to_snomedct_us(self):
        with patch(
            "departments.medicine.umls_client.search_umls",
            return_value=[{"code": "424754009", "description": "Fever"}],
        ) as mock:
            from departments.medicine.umls_client import search_snomed_live

            result = search_snomed_live("fever")
        mock.assert_called_once_with("fever", sab="SNOMEDCT_US", max_results=20)
        assert result[0]["code"] == "424754009"

    def test_search_loinc_live_routes_to_lnc(self):
        with patch(
            "departments.medicine.umls_client.search_umls",
            return_value=[{"code": "1558-6", "description": "Fasting glucose"}],
        ) as mock:
            from departments.medicine.umls_client import search_loinc_live

            result = search_loinc_live("glucose")
        mock.assert_called_once_with("glucose", sab="LNC", max_results=20)
        assert result[0]["code"] == "1558-6"

    # -- seed datasets -------------------------------------------------------

    def test_snomed_seed_dataset_has_required_east_africa_codes(self):
        from departments.medicine.umls_client import get_core_snomed_seed_dataset

        seeds = get_core_snomed_seed_dataset()
        codes = {s["code"] for s in seeds}
        assert len(seeds) >= 40
        assert "424754009" in codes  # Fever
        assert "61462000" in codes  # Malaria
        assert "56717001" in codes  # Tuberculosis
        assert "86406008" in codes  # HIV
        for s in seeds:
            assert s.get("code") and s.get("description")

    def test_loinc_seed_dataset_has_required_vitals_and_panels(self):
        from departments.medicine.umls_client import get_core_loinc_seed_dataset

        seeds = get_core_loinc_seed_dataset()
        codes = {s["code"] for s in seeds}
        assert len(seeds) >= 38
        assert "8302-2" in codes  # Body height
        assert "8480-6" in codes  # Systolic BP
        assert "1558-6" in codes  # Fasting glucose
        assert "43012-4" in codes  # HIV rapid test
        for s in seeds:
            assert s.get("code") and s.get("description")


# ===========================================================================
# 3. ICD-10 Importer
# ===========================================================================
# NOTE: icd10_importer.import_from_who_api() does a local import of
# walk_icd10_tree from the who_icd_client module:
#   `from departments.medicine.who_icd_client import walk_icd10_tree`
# Therefore the correct mock target is departments.medicine.who_icd_client,
# NOT departments.medicine.icd10_importer (which never binds that name at
# module level).


class TestIcd10Importer:
    """departments/medicine/icd10_importer.py"""

    def test_import_from_who_api_inserts_new_codes(self, app):
        tree_codes = [
            ("A01.0", "Typhoid fever", "Infectious", "A00-A09"),
            ("B50", "P. falciparum malaria", "Parasitic", "B50-B54"),
            ("J18.9", "Pneumonia unspecified", "Respiratory", "J09-J18"),
        ]
        with app.app_context():
            with patch(
                "departments.medicine.who_icd_client.walk_icd10_tree",
                return_value=iter(tree_codes),
            ):
                from departments.medicine.icd10_importer import import_from_who_api

                count = import_from_who_api("2019")

            assert count == 3
            from departments.models.terminology import ICD10Code

            assert ICD10Code.query.count() == 3
            row = ICD10Code.query.filter_by(code="A01.0").first()
            assert row.chapter == "Infectious"
            assert row.block == "A00-A09"

    def test_import_from_who_api_upserts_existing_row(self, app):
        """Second run with updated description must not create a duplicate row."""
        with app.app_context():
            with patch(
                "departments.medicine.who_icd_client.walk_icd10_tree",
                return_value=iter([("A01.0", "Typhoid (old)", "Inf", "A00")]),
            ):
                from departments.medicine.icd10_importer import import_from_who_api

                import_from_who_api("2019")

            with patch(
                "departments.medicine.who_icd_client.walk_icd10_tree",
                return_value=iter([("A01.0", "Typhoid (updated)", "Inf", "A00")]),
            ):
                import_from_who_api("2019")

            from departments.models.terminology import ICD10Code

            assert ICD10Code.query.count() == 1
            assert (
                ICD10Code.query.filter_by(code="A01.0").first().description
                == "Typhoid (updated)"
            )

    def test_import_from_who_api_handles_250_codes_in_two_batches(self, app):
        """250 codes with batch_size=200 → 2 flush cycles, all rows committed."""
        tree_codes = [(f"X{i:03d}", f"Disease {i}", "Ch", "Blk") for i in range(250)]
        with app.app_context():
            with patch(
                "departments.medicine.who_icd_client.walk_icd10_tree",
                return_value=iter(tree_codes),
            ):
                from departments.medicine.icd10_importer import import_from_who_api

                count = import_from_who_api("2019")

            assert count == 250
            from departments.models.terminology import ICD10Code

            assert ICD10Code.query.count() == 250

    def test_import_from_who_api_rolls_back_and_reraises_on_walk_error(self, app):
        def bad_walk(release):
            yield "A01.0", "Typhoid", "Inf", "A00"
            raise RuntimeError("WHO API exploded mid-walk")

        with app.app_context(), patch(
            "departments.medicine.who_icd_client.walk_icd10_tree",
            side_effect=bad_walk,
        ):
            from departments.medicine.icd10_importer import import_from_who_api

            with pytest.raises(RuntimeError, match="WHO API exploded mid-walk"):
                import_from_who_api("2019")

    # -- load_icd10_from_csv -------------------------------------------------

    def test_load_icd10_from_csv_parses_all_columns(self, tmp_path):
        csv_file = tmp_path / "icd10.csv"
        csv_file.write_text(
            "CODE,DESCRIPTION,CHAPTER,BLOCK\n"
            "A01.0,Typhoid fever,Infectious,A00-A09\n"
            "B50,Malaria,Parasitic,B50-B54\n"
        )
        from departments.medicine.icd10_importer import load_icd10_from_csv

        codes = load_icd10_from_csv(str(csv_file))
        assert len(codes) == 2
        assert codes[0] == {
            "code": "A01.0",
            "description": "Typhoid fever",
            "chapter": "Infectious",
            "block": "A00-A09",
        }

    def test_import_icd10_codes_missing_file_returns_zero(self, app):
        from departments.medicine.icd10_importer import import_icd10_codes

        assert import_icd10_codes("/nonexistent/path/icd10.csv") == 0

    def test_import_icd10_codes_from_csv_file(self, app, tmp_path):
        csv_file = tmp_path / "icd10.csv"
        csv_file.write_text(
            "CODE,DESCRIPTION,CHAPTER,BLOCK\n"
            "A01.0,Typhoid fever,Infectious,A00-A09\n"
            "B50,Malaria,Parasitic,B50-B54\n"
        )
        from departments.medicine.icd10_importer import import_icd10_codes

        count = import_icd10_codes(str(csv_file))
        assert count == 2
        from departments.models.terminology import ICD10Code

        assert ICD10Code.query.count() == 2


# ===========================================================================
# 4. SNOMED CT Importer
# ===========================================================================


class TestSnomedImporter:
    """departments/medicine/snomed_importer.py"""

    def test_import_from_umls_api_seed_only_populates_db(self, app):
        """fetch_live_api=False → only the 43-entry seed dataset is loaded."""
        with app.app_context():
            from departments.medicine.snomed_importer import import_from_umls_api

            count = import_from_umls_api(fetch_live_api=False)

            assert count >= 40
            from departments.models.terminology import SnomedCode

            assert SnomedCode.query.count() >= 40

    def test_import_from_umls_api_live_results_supplement_seed(self, app):
        """A novel code from the live API is merged with the seed codes."""
        extra = [{"code": "99999999", "description": "Test Finding", "cui": "C9999"}]
        with app.app_context():
            with patch(
                "departments.medicine.snomed_importer.search_snomed_live",
                return_value=extra,
            ):
                from departments.medicine.snomed_importer import import_from_umls_api

                count = import_from_umls_api(fetch_live_api=True)

            assert count >= 40
            from departments.models.terminology import SnomedCode

            row = SnomedCode.query.filter_by(code="99999999").first()
            assert row is not None
            assert row.description == "Test Finding"

    def test_import_from_umls_api_upserts_changed_description(self, app):
        with app.app_context():
            from departments.medicine.snomed_importer import import_from_umls_api

            import_from_umls_api(fetch_live_api=False)

            updated = [{"code": "424754009", "description": "Fever (updated)"}]
            with patch(
                "departments.medicine.snomed_importer.get_core_snomed_seed_dataset",
                return_value=updated,
            ):
                import_from_umls_api(fetch_live_api=False)

            from departments.models.terminology import SnomedCode

            row = SnomedCode.query.filter_by(code="424754009").first()
            assert row.description == "Fever (updated)"

    def test_import_from_umls_api_skips_failed_live_term(self, app):
        """A network error for one search term must not abort the whole import."""

        def flaky(term, max_results=20):
            raise requests.ConnectionError("network down")

        with app.app_context(), patch(
            "departments.medicine.snomed_importer.search_snomed_live",
            side_effect=flaky,
        ):
            from departments.medicine.snomed_importer import import_from_umls_api

            count = import_from_umls_api(fetch_live_api=True)

        assert count >= 40  # seed still loaded despite every live call failing

    # -- load_snomed_from_csv / import_snomed_codes --------------------------

    def test_load_snomed_from_csv_parses_rows(self, tmp_path):
        csv_file = tmp_path / "snomed.csv"
        csv_file.write_text(
            "CODE,DESCRIPTION\n424754009,Fever\n38341003,Hypertension\n"
        )
        from departments.medicine.snomed_importer import load_snomed_from_csv

        codes = load_snomed_from_csv(str(csv_file))
        assert len(codes) == 2
        assert codes[0] == {"code": "424754009", "description": "Fever"}

    def test_import_snomed_codes_falls_back_to_umls_when_csv_missing(self, app):
        with app.app_context(), patch(
            "departments.medicine.snomed_importer.import_from_umls_api",
            return_value=42,
        ) as mock_api:
            from departments.medicine.snomed_importer import import_snomed_codes

            result = import_snomed_codes("/nonexistent/snomed.csv")
        mock_api.assert_called_once_with(fetch_live_api=True)
        assert result == 42

    def test_import_snomed_codes_from_csv_file(self, app, tmp_path):
        csv_file = tmp_path / "snomed.csv"
        csv_file.write_text("CODE,DESCRIPTION\n424754009,Fever\n")
        with app.app_context():
            from departments.medicine.snomed_importer import import_snomed_codes

            count = import_snomed_codes(str(csv_file))
            assert count == 1
            from departments.models.terminology import SnomedCode

            assert SnomedCode.query.filter_by(code="424754009").count() == 1


# ===========================================================================
# 5. LOINC Importer
# ===========================================================================


class TestLoincImporter:
    """departments/medicine/loinc_importer.py"""

    def test_import_from_umls_api_seed_only_populates_db(self, app):
        with app.app_context():
            from departments.medicine.loinc_importer import import_from_umls_api

            count = import_from_umls_api(fetch_live_api=False)

            assert count >= 38
            from departments.models.terminology import LoincCode

            assert LoincCode.query.count() >= 38

    def test_import_from_umls_api_live_results_supplement_seed(self, app):
        extra = [{"code": "99999-9", "description": "Novel Lab Test", "cui": "C9999"}]
        with app.app_context():
            with patch(
                "departments.medicine.loinc_importer.search_loinc_live",
                return_value=extra,
            ):
                from departments.medicine.loinc_importer import import_from_umls_api

                count = import_from_umls_api(fetch_live_api=True)

            assert count >= 38
            from departments.models.terminology import LoincCode

            assert LoincCode.query.filter_by(code="99999-9").count() == 1

    def test_import_from_umls_api_skips_failed_live_term(self, app):
        with app.app_context(), patch(
            "departments.medicine.loinc_importer.search_loinc_live",
            side_effect=requests.ConnectionError("loinc timeout"),
        ):
            from departments.medicine.loinc_importer import import_from_umls_api

            count = import_from_umls_api(fetch_live_api=True)
        assert count >= 38

    def test_import_from_umls_api_upserts_changed_description(self, app):
        with app.app_context():
            from departments.medicine.loinc_importer import import_from_umls_api

            import_from_umls_api(fetch_live_api=False)

            updated = [{"code": "8302-2", "description": "Body height (cm)"}]
            with patch(
                "departments.medicine.loinc_importer.get_core_loinc_seed_dataset",
                return_value=updated,
            ):
                import_from_umls_api(fetch_live_api=False)

            from departments.models.terminology import LoincCode

            assert (
                LoincCode.query.filter_by(code="8302-2").first().description
                == "Body height (cm)"
            )

    # -- load_loinc_from_csv / import_loinc_codes ----------------------------

    def test_load_loinc_from_csv_parses_rows(self, tmp_path):
        csv_file = tmp_path / "loinc.csv"
        csv_file.write_text(
            "LOINC_NUM,LONG_COMMON_NAME\n8302-2,Body height\n8480-6,Systolic BP\n"
        )
        from departments.medicine.loinc_importer import load_loinc_from_csv

        codes = load_loinc_from_csv(str(csv_file))
        assert len(codes) == 2
        assert codes[0] == {"code": "8302-2", "description": "Body height"}

    def test_import_loinc_codes_falls_back_to_umls_when_csv_missing(self, app):
        with app.app_context(), patch(
            "departments.medicine.loinc_importer.import_from_umls_api",
            return_value=41,
        ) as mock_api:
            from departments.medicine.loinc_importer import import_loinc_codes

            result = import_loinc_codes("/nonexistent/loinc.csv")
        mock_api.assert_called_once_with(fetch_live_api=True)
        assert result == 41

    def test_import_loinc_codes_from_csv_file(self, app, tmp_path):
        csv_file = tmp_path / "loinc.csv"
        csv_file.write_text("LOINC_NUM,LONG_COMMON_NAME\n8302-2,Body height\n")
        with app.app_context():
            from departments.medicine.loinc_importer import import_loinc_codes

            count = import_loinc_codes(str(csv_file))
            assert count == 1
            from departments.models.terminology import LoincCode

            assert LoincCode.query.filter_by(code="8302-2").count() == 1


# ===========================================================================
# 6. Celery Nightly Sync Tasks
# ===========================================================================
# Strategy: patch the importer functions AT THEIR CALL SITE inside the task body.
# Each task does `from app import app` and opens its own app_context(); we let
# the real Flask app (from conftest) run and only stub the heavy importer calls.


class TestTerminologyCeleryTasks:
    """departments/tasks.py — sync_icd10_codes, sync_snomed_codes, sync_loinc_codes"""

    def test_sync_icd10_codes_returns_ok_status(self, app):
        # Patch the importer at its call-site module (the tasks module imports it locally)
        with patch(
            "departments.medicine.icd10_importer.import_from_who_api",
            return_value=14200,
        ):
            from departments.tasks import sync_icd10_codes

            result = sync_icd10_codes()
        assert result == {"status": "ok", "codes_upserted": 14200}

    def test_sync_snomed_codes_returns_ok_status(self, app):
        with patch(
            "departments.medicine.snomed_importer.import_from_umls_api",
            return_value=440,
        ):
            from departments.tasks import sync_snomed_codes

            result = sync_snomed_codes()
        assert result == {"status": "ok", "codes_upserted": 440}

    def test_sync_loinc_codes_returns_ok_status(self, app):
        with patch(
            "departments.medicine.loinc_importer.import_from_umls_api",
            return_value=95,
        ):
            from departments.tasks import sync_loinc_codes

            result = sync_loinc_codes()
        assert result == {"status": "ok", "codes_upserted": 95}

    def test_sync_icd10_codes_propagates_importer_exception(self, app):
        with patch(
            "departments.medicine.icd10_importer.import_from_who_api",
            side_effect=RuntimeError("WHO API down"),
        ):
            from departments.tasks import sync_icd10_codes

            with pytest.raises(RuntimeError, match="WHO API down"):
                sync_icd10_codes()

    def test_sync_snomed_codes_propagates_importer_exception(self, app):
        with patch(
            "departments.medicine.snomed_importer.import_from_umls_api",
            side_effect=RuntimeError("UMLS unreachable"),
        ):
            from departments.tasks import sync_snomed_codes

            with pytest.raises(RuntimeError, match="UMLS unreachable"):
                sync_snomed_codes()

    def test_sync_loinc_codes_propagates_importer_exception(self, app):
        with patch(
            "departments.medicine.loinc_importer.import_from_umls_api",
            side_effect=RuntimeError("LOINC timeout"),
        ):
            from departments.tasks import sync_loinc_codes

            with pytest.raises(RuntimeError, match="LOINC timeout"):
                sync_loinc_codes()
