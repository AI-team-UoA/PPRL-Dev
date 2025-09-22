import hashlib

def pseudonymize(s: str, salt: str = "") -> str:
    salted = (salt + s).encode("utf-8")
    return hashlib.sha256(salted).hexdigest()