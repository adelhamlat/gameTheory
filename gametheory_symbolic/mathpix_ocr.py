import json
import requests
import re

_CODE_FENCE = re.compile(r"^\s*```.*$")

MATHPIX_API = "https://api.mathpix.com/v3/text"
APP_ID  = "monad_571e3f_bdc32f"
APP_KEY = "fc605c7b50ccf427baa28a8adbd59c094ec9d4b1287792ee79597f30ad7115f2"


HEADERS_JSON = {
    "app_id": APP_ID,
    "app_key": APP_KEY,
    "Content-Type": "application/json",
}

HEADERS_FORM = {
    "app_id": APP_ID,
    "app_key": APP_KEY,
}

# Options fréquentes : renvoyer LaTeX stylé et texte ; forcer nettoyage des espaces.
DEFAULT_OPTIONS = {
    "formats": ["latex_styled", "text", "data"],   # vous pouvez réduire cette liste
    "rm_spaces": True,
    # Exemple: "math_inline_delimiters": ["$", "$"]
}

def ocr_image_file(path: str, options: dict | None = None) -> dict:
    """Envoie un fichier image local à v3/text (multipart/form-data)."""
    opts = options or DEFAULT_OPTIONS
    with open(path, "rb") as f:
        resp = requests.post(
            MATHPIX_API,
            headers=HEADERS_FORM,
            files={"file": f},
            data={"options_json": json.dumps(opts)},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()

def ocr_image_url(url: str, options: dict | None = None) -> dict:
    """Envoie une URL d'image à v3/text (JSON)."""
    opts = options or DEFAULT_OPTIONS
    payload = {"src": url, **opts}
    resp = requests.post(
        MATHPIX_API,
        headers=HEADERS_JSON,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()

def extract_equations(mathpix_response: dict) -> list[str]:
    """
    Extrait uniquement les lignes contenant des équations (avec '=') 
    du champ 'text' de la réponse Mathpix.
    """
    text = mathpix_response.get("text", "")
    lines = [l.strip() for l in text.splitlines()]
    # Garde uniquement les lignes qui ressemblent à des équations
    equations = [l for l in lines if "=" in l and not l.startswith("#") and len(l) > 3]
    return equations



def extract_single_equation(mathpix_response: dict) -> str | None:
    """
    Retourne UNE seule équation depuis le champ 'text' de Mathpix.
    - Ignore les lignes vides, titres '#', et fences Markdown ```.
    - Ne garde que la première ligne ayant un '=' avec une LHS et RHS non vides.
    """
    text = (mathpix_response or {}).get("text", "") or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for l in lines:
        if _CODE_FENCE.match(l):     # ignore ``` fences
            continue
        if l.startswith("#"):        # ignore titres
            continue
        if "=" not in l:
            continue
        left, right = l.split("=", 1)
        if left.strip() and right.strip():
            return l.strip()
    return None
