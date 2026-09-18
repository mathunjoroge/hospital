"""
departments/medicine/who_icd_client.py
────────────────────────────────────────
WHO ICD-10 API client.

Handles OAuth2 client_credentials token lifecycle and exposes two public
functions used by the rest of the system:

  walk_icd10_tree(release)        → generator of (code, title, chapter, block)
  search_icd10_live(query, release) → list[dict] from the WHO search API

Token caching:
  The WHO token endpoint returns a 3600-second Bearer JWT. We cache it
  in-process and refresh 60 s before expiry so we never make an extra
  round-trip for every leaf-node fetch during the bulk walk.

Rate limiting:
  WHO enforces roughly 2 requests/second. We stay safely under that with a
  0.55 s inter-request delay during tree walking. On transient errors we
  use truncated exponential back-off (max 3 retries).
"""

import logging
import os
import time
from threading import Lock

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
_API_BASE = (
    "https://id.who.int/icd"  # Always HTTPS — avoids HTTP→HTTPS redirect overhead
)
_DEFAULT_RELEASE = os.getenv("WHO_ICD_API_RELEASE", "2019")

# Shared headers required by every WHO ICD API call
_COMMON_HEADERS = {
    "Accept": "application/json",
    "API-Version": "v2",
    "Accept-Language": "en",
}

# Delay between consecutive tree-walk requests (seconds).
# WHO rate limit is ~2 req/s; 0.55 s gives comfortable headroom.
_TREE_WALK_DELAY = 0.55

# ---------------------------------------------------------------------------
# Token cache (in-process, thread-safe)
# ---------------------------------------------------------------------------
_token_lock = Lock()
_cached_token: str | None = None
_token_expires_at: float = 0.0  # epoch seconds


def _get_bearer_token() -> str:
    """Return a valid Bearer token, refreshing it when near expiry."""
    global _cached_token, _token_expires_at

    client_id = os.getenv("WHO_ICD_CLIENT_ID", "")
    client_secret = os.getenv("WHO_ICD_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        raise RuntimeError(
            "WHO_ICD_CLIENT_ID and WHO_ICD_CLIENT_SECRET must be set in .env "
            "(see DECISIONS_PENDING #4)."
        )

    with _token_lock:
        # Refresh 60 s before actual expiry to avoid mid-walk token failures
        if _cached_token and time.time() < _token_expires_at - 60:
            return _cached_token

        logger.info("Fetching new WHO ICD API Bearer token …")
        resp = requests.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "icdapi_access",
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        _cached_token = payload["access_token"]
        _token_expires_at = time.time() + payload.get("expires_in", 3600)
        logger.info(
            "WHO ICD API token acquired (expires in %ds).",
            payload.get("expires_in", 3600),
        )
        return _cached_token


def _force_https(url: str) -> str:
    """Upgrade http:// WHO API child URLs to https:// — avoids a 301 redirect per node."""
    if url.startswith("http://"):
        return "https://" + url[7:]
    return url


def _api_get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    """
    GET a WHO ICD API URL with automatic token injection and retry logic.

    Raises requests.HTTPError on permanent failure.
    """
    url = _force_https(url)
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            token = _get_bearer_token()
            headers = {**_COMMON_HEADERS, "Authorization": f"Bearer {token}"}
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                # Token rejected — force refresh on next attempt
                global _cached_token
                _cached_token = None
            last_exc = exc
        except requests.RequestException as exc:
            last_exc = exc

        if attempt < retries - 1:
            wait = delay * (2**attempt)
            logger.warning(
                "WHO API request failed (attempt %d/%d) — retrying in %.1fs: %s",
                attempt + 1,
                retries,
                wait,
                last_exc,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"WHO API request failed after {retries} attempts: {url}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------
def walk_icd10_tree(release: str = _DEFAULT_RELEASE):
    """
    Generator that recursively walks the full WHO ICD-10 classification tree
    for *release* (e.g. '2019') and yields one tuple per leaf code:

        (code: str, title: str, chapter: str, block: str)

    Non-leaf nodes (chapters, blocks) that carry their own code are also
    yielded so the DB is complete for lookup purposes.

    The walk is depth-first. Expect ~14,000–14,500 items for the 2019 release.
    """
    root_url = f"{_API_BASE}/release/10/{release}"
    root = _api_get(root_url)
    chapter_urls = root.get("child", [])

    for ch_url in chapter_urls:
        ch_data = _api_get(ch_url)
        time.sleep(_TREE_WALK_DELAY)

        chapter_title = ch_data.get("title", {}).get("@value", "")
        chapter_code = ch_data.get("code", "")

        # Yield the chapter node itself if it has a usable code
        if chapter_code and chapter_code not in (
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
        ):
            yield chapter_code, chapter_title, chapter_title, ""

        for block_url in ch_data.get("child", []):
            block_data = _api_get(block_url)
            time.sleep(_TREE_WALK_DELAY)

            block_title = block_data.get("title", {}).get("@value", "")
            block_code = block_data.get("code", "")

            # Yield the block node
            if block_code:
                yield block_code, block_title, chapter_title, block_title

            yield from _walk_children(
                block_data.get("child", []),
                chapter_title=chapter_title,
                block_title=block_title,
            )


def _walk_children(child_urls: list[str], chapter_title: str, block_title: str):
    """Recursively yield (code, title, chapter, block) for all descendant nodes."""
    for child_url in child_urls:
        child_data = _api_get(child_url)
        time.sleep(_TREE_WALK_DELAY)

        code = child_data.get("code", "")
        title = child_data.get("title", {}).get("@value", "")

        if code and title:
            yield code, title, chapter_title, block_title

        grandchildren = child_data.get("child", [])
        if grandchildren:
            yield from _walk_children(
                grandchildren, chapter_title=chapter_title, block_title=block_title
            )


# ---------------------------------------------------------------------------
# Live search (typeahead)
# ---------------------------------------------------------------------------
def search_icd10_live(query: str, release: str = _DEFAULT_RELEASE) -> list[dict]:
    """
    Call the WHO ICD-10 search endpoint and return a normalised list of dicts:

        [{"code": "B50", "description": "Plasmodium falciparum malaria", "category": "..."}, ...]

    Used as a live fallback when the local DB has fewer than 100 codes.
    """
    if not query:
        return []

    search_url = f"{_API_BASE}/release/10/{release}/search"
    try:
        data = _api_get(
            search_url,
            params={
                "q": query,
                "useFlexisearch": "false",
                "flatResults": "true",
            },
        )
        destination_entities = data.get("destinationEntities", [])
        results = []
        for entity in destination_entities[:20]:
            code = entity.get("theCode", "")
            desc = entity.get("title", "")
            # title is sometimes a list of strings; flatten if needed
            if isinstance(desc, list):
                desc = " | ".join(desc)
            if code:
                results.append(
                    {
                        "code": code,
                        "description": desc,
                        "category": entity.get("chapter", ""),
                    }
                )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("WHO live ICD-10 search failed for %r: %s", query, exc)
        return []
