import re

PLATE_REGEX = re.compile(
    r'^[A-Z]{1,3}[A-Z0-9]{4,6}$'
)

def normalize_plate(text: str) -> str:
    return (
        text.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .strip()
    )

def looks_like_plate(text: str) -> bool:
    return bool(PLATE_REGEX.match(text))