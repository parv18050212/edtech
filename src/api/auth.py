import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

SUPABASE_PROJECT_REF = "aedsgktxvmddarqurbhi"
ISSUER = f"https://{SUPABASE_PROJECT_REF}.supabase.co/auth/v1"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
AUDIENCE = "authenticated"

_jwk_client = PyJWKClient(JWKS_URL)


def _decode(token: str, key, issuer: str, audience: str, algorithms: list[str]) -> dict:
    return jwt.decode(token, key, algorithms=algorithms, issuer=issuer, audience=audience)


def verify_token(token: str) -> dict:
    signing_key = _jwk_client.get_signing_key_from_jwt(token)
    return _decode(token, signing_key.key, ISSUER, AUDIENCE, ["ES256"])


def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ")
    try:
        payload = verify_token(token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["sub"]
