import pathlib
import tempfile

import requests


def osf_fetch(slug: str) -> pathlib.Path:
    """Download a file from OSF by its slug, caching it under the system
    temp directory, and return the path to the cached file."""
    cache_path = pathlib.Path(tempfile.gettempdir()) / slug
    url = f"https://osf.io/{slug}/download"
    if not cache_path.exists():
        print(f"downloading {url} -> {cache_path}")
        resp = requests.get(url, allow_redirects=True, timeout=180)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    else:
        print(f"reusing cached {cache_path}")
    print(f"size: {cache_path.stat().st_size} bytes")
    return cache_path
