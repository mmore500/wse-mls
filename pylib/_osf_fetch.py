import pathlib
import tempfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def osf_fetch(slug: str) -> pathlib.Path:
    """Download a file from OSF by its slug, caching it under the system
    temp directory, and return the path to the cached file.

    Transient server errors (e.g., 502 Bad Gateway) are retried with
    exponential backoff."""
    cache_path = pathlib.Path(tempfile.gettempdir()) / slug
    url = f"https://osf.io/{slug}/download"
    if not cache_path.exists():
        print(f"downloading {url} -> {cache_path}")
        retry = Retry(
            total=6,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            resp = session.get(url, allow_redirects=True, timeout=180)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    else:
        print(f"reusing cached {cache_path}")
    print(f"size: {cache_path.stat().st_size} bytes")
    return cache_path
