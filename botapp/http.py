import os
import certifi
import tempfile
import requests

_CA_PATH = None

def _get_ca_bundle_path() -> str:
    global _CA_PATH
    base = certifi.where()
    extra = os.getenv('EXTRA_CA_BUNDLE')
    if not extra or not os.path.isfile(extra):
        return base
    if _CA_PATH:
        return _CA_PATH
    try:
        combined = os.path.join(tempfile.gettempdir(), 'combined_cacert.pem')
        with open(base, 'rb') as b, open(extra, 'rb') as e, open(combined, 'wb') as out:
            out.write(b.read())
            out.write(b'\n')
            out.write(e.read())
        _CA_PATH = combined
        return combined
    except Exception:
        return base

def set_ca_env() -> None:
    p = _get_ca_bundle_path()
    os.environ['SSL_CERT_FILE'] = p
    os.environ['REQUESTS_CA_BUNDLE'] = p
    os.environ['CURL_CA_BUNDLE'] = p

def post_json(url: str, data: dict, headers: dict, timeout: int = 60) -> requests.Response:
    s = requests.Session()
    s.trust_env = False
    return s.post(url, json=data, headers=headers, timeout=timeout, verify=_get_ca_bundle_path())

def get_stream(url: str, timeout: int = 120) -> requests.Response:
    s = requests.Session()
    s.trust_env = False
    return s.get(url, stream=True, timeout=timeout, verify=_get_ca_bundle_path())


