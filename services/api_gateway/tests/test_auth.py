import pytest
from fastapi import HTTPException

from app.core import security
from app.core.config import settings
from app.core.security import create_access_token, get_tenant


def test_default_tenant_when_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_auth", False)
    assert get_tenant() == settings.default_tenant


def test_api_key_resolves_tenant(monkeypatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "auth_api_keys", "keyA:acme,keyB:globex")
    assert get_tenant(x_api_key="keyA") == "acme"
    assert get_tenant(x_api_key="keyB") == "globex"


def test_bad_api_key_rejected(monkeypatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "auth_api_keys", "keyA:acme")
    with pytest.raises(HTTPException) as exc:
        get_tenant(x_api_key="nope")
    assert exc.value.status_code == 401


def test_missing_credentials_rejected(monkeypatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "auth_api_keys", "keyA:acme")
    with pytest.raises(HTTPException) as exc:
        get_tenant()
    assert exc.value.status_code == 401


def test_jwt_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    token = create_access_token("globex")
    assert get_tenant(authorization=f"Bearer {token}") == "globex"


def test_invalid_jwt_rejected(monkeypatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    with pytest.raises(HTTPException) as exc:
        get_tenant(authorization="Bearer not-a-real-token")
    assert exc.value.status_code == 401
