"""Tokens minted on a host whose clock is not this host's.

Found on the live two-machine install, 2026-09-15: the control root ran
0.5 s ahead of a worker, and with zero leeway PyJWT refused every token
the root minted in the first half of each second as "not yet valid
(iat)". The root's probes 401'd on a loop, the node was listed `down`,
and nothing said why. Five minutes is the Kerberos/Entra convention;
anything inside it is accepted, and anything a token was issued in this
host's future by is logged so a wrong clock is visible before it grows
past the leeway.
"""

from __future__ import annotations

import logging
import time

import jwt
import pytest

from eugene_plexus_gateway import security

KEY = bytes(range(32))


def _token(*, iat: int, exp: int | None = None) -> str:
    claims = {"sub": "control", "aud": "service:control", "iat": iat, "exp": exp or iat + 3600}
    return jwt.encode(claims, KEY, algorithm="HS256")


def test_a_token_from_a_clock_half_a_second_ahead_is_accepted() -> None:
    # The live case: the issuer's integer second is one past this host's.
    iat = int(time.time()) + 1
    payload = security.decode_token(token=_token(iat=iat), signing_key=KEY)
    assert payload.iat == iat


def test_skew_inside_the_leeway_is_accepted() -> None:
    iat = int(time.time()) + security.CLOCK_SKEW_LEEWAY_SECONDS - 5
    assert security.decode_token(token=_token(iat=iat), signing_key=KEY).sub == "control"


def test_skew_past_the_leeway_is_still_refused() -> None:
    iat = int(time.time()) + security.CLOCK_SKEW_LEEWAY_SECONDS + 5
    with pytest.raises(jwt.ImmatureSignatureError):
        security.decode_token(token=_token(iat=iat), signing_key=KEY)


def test_expiry_gets_the_same_leeway() -> None:
    now = int(time.time())
    inside = _token(iat=now - 3600, exp=now - security.CLOCK_SKEW_LEEWAY_SECONDS + 5)
    assert security.decode_token(token=inside, signing_key=KEY).sub == "control"
    outside = _token(iat=now - 3600, exp=now - security.CLOCK_SKEW_LEEWAY_SECONDS - 5)
    with pytest.raises(jwt.ExpiredSignatureError):
        security.decode_token(token=outside, signing_key=KEY)


def test_a_future_issued_token_is_logged_once_a_minute(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(security, "_last_skew_warning", 0.0)
    now = int(time.time())
    with caplog.at_level(logging.WARNING, logger=security.__name__):
        security.decode_token(token=_token(iat=now + 30), signing_key=KEY)
        security.decode_token(token=_token(iat=now + 31), signing_key=KEY)
    skew = [r for r in caplog.records if "in this host's future" in r.getMessage()]
    assert len(skew) == 1, [r.getMessage() for r in caplog.records]
    message = skew[0].getMessage()
    ahead = float(message.split("issued ")[1].split(" s")[0])
    assert 28.0 < ahead <= 30.0 and "300 s" in message, message


def test_an_ordinary_token_is_not_logged_as_skew(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(security, "_last_skew_warning", 0.0)
    with caplog.at_level(logging.WARNING, logger=security.__name__):
        security.decode_token(token=_token(iat=int(time.time()) - 1), signing_key=KEY)
    assert not [r for r in caplog.records if "future" in r.getMessage()]
