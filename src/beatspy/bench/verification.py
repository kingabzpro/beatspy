"""Owner attestations bind replayable artifacts to a trusted execution and release."""

import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from ..artifacts import atomic_json, sha256
from .validation import read_json, validate_run


def code_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.suffix in (".py", ".toml")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    for name in ("pyproject.toml", "uv.lock"):
        path = root.parent.parent / name
        if path.exists():
            digest.update(name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def key_id(key: Ed25519PublicKey) -> str:
    return hashlib.sha256(key.public_bytes_raw()).hexdigest()


def sign_result(run_dir: Path, private_pem: bytes, execution: str) -> dict:
    """Called only after the controlled runner exports and replays its own new run."""
    meta = validate_run(run_dir)
    if meta.get("code_sha256") != code_digest():
        raise ValueError("run code integrity does not match the trusted runner")
    key = serialization.load_pem_private_key(private_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("signing key must be an Ed25519 private PEM key")
    payload = {
        "schema_version": 1,
        "algorithm": "Ed25519",
        "verification_id": str(uuid4()),
        "run_id": meta["run_id"],
        "run_sha256": sha256(run_dir / "run.json"),
        "beatspy_version": meta["beatspy_version"],
        "protocol_version": meta["protocol_version"],
        "code_sha256": meta["code_sha256"],
        "source_revision": meta["source_revision"],
        "execution": execution,
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "key_id": key_id(key.public_key()),
    }
    envelope = {"payload": payload, "signature": base64.b64encode(key.sign(canonical(payload))).decode()}
    atomic_json(run_dir / "verification.json", envelope)
    return envelope


def verify_result(run_dir: Path, public_pem: bytes, *, release: tuple[str, str] | None = None) -> dict:
    envelope = read_json(run_dir / "verification.json")
    payload = envelope["payload"]
    key = serialization.load_pem_public_key(public_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("verification key must be an Ed25519 public PEM key")
    try:
        key.verify(base64.b64decode(envelope["signature"], validate=True), canonical(payload))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("invalid result signature") from exc
    if payload.get("algorithm") != "Ed25519" or payload.get("schema_version") != 1:
        raise ValueError("unsupported verification format")
    if payload.get("key_id") != key_id(key) or UUID(payload["verification_id"]).version != 4:
        raise ValueError("invalid verification identity")
    meta = read_json(run_dir / "run.json")
    for field in ("run_id", "beatspy_version", "protocol_version", "code_sha256", "source_revision"):
        if payload.get(field) != meta.get(field) or payload.get(field) is None:
            raise ValueError(f"verification mismatch: {field}")
    if payload["run_sha256"] != sha256(run_dir / "run.json"):
        raise ValueError("verification mismatch: run hash")
    if release is not None and (payload["beatspy_version"], payload["code_sha256"]) != release:
        raise ValueError("unapproved benchmark release or code integrity")
    return payload
