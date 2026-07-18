import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from api.auth import _decode

ISSUER = "https://example.supabase.co/auth/v1"
AUDIENCE = "authenticated"


def _make_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    return private_key, private_key.public_key()


def _make_token(private_key, **claim_overrides):
    now = int(time.time())
    claims = {
        "sub": "user-123",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": now + 3600,
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, private_key, algorithm="ES256")


def test_decode_accepts_valid_token():
    private_key, public_key = _make_keypair()
    token = _make_token(private_key)
    payload = _decode(token, public_key, ISSUER, AUDIENCE, ["ES256"])
    assert payload["sub"] == "user-123"


def test_decode_rejects_wrong_issuer():
    private_key, public_key = _make_keypair()
    token = _make_token(private_key, iss="https://evil.example.com/auth/v1")
    with pytest.raises(jwt.InvalidIssuerError):
        _decode(token, public_key, ISSUER, AUDIENCE, ["ES256"])


def test_decode_rejects_wrong_audience():
    private_key, public_key = _make_keypair()
    token = _make_token(private_key, aud="something-else")
    with pytest.raises(jwt.InvalidAudienceError):
        _decode(token, public_key, ISSUER, AUDIENCE, ["ES256"])


def test_decode_rejects_expired_token():
    private_key, public_key = _make_keypair()
    token = _make_token(private_key, exp=int(time.time()) - 10)
    with pytest.raises(jwt.ExpiredSignatureError):
        _decode(token, public_key, ISSUER, AUDIENCE, ["ES256"])


def test_decode_rejects_token_signed_by_a_different_key():
    _, real_public_key = _make_keypair()
    other_private_key, _ = _make_keypair()
    token = _make_token(other_private_key)  # signed by a DIFFERENT key
    with pytest.raises(jwt.InvalidSignatureError):
        _decode(token, real_public_key, ISSUER, AUDIENCE, ["ES256"])
