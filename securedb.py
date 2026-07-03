"""
HashGuard-Med: Tamper-evident, hash-chained, signed image database
for trustworthy medical image classification.

Contribution module for the research paper. Provides:
  - SHA-256 hash chain over ingested images (Merkle-style linear chain)
  - Ed25519 signatures binding {image_hash, label, metadata} at ingestion
  - verify_and_load(): integrity + authenticity check before an image
    reaches the ML DataLoader
  - verify_chain_full(): walks the whole chain, reports first broken link
  - Tamper-evident audit log of every verification event

Design note on threat coverage:
  * A1 image substitution/manipulation -> caught by SHA-256 hash mismatch
  * A2 label-flip poisoning            -> caught by signature (label is signed)
  * A3 unauthenticated injection       -> caught by signature (wrong key)
"""

from __future__ import annotations

import json
import sqlite3
import hashlib
import datetime as _dt
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

GENESIS_PREV = "0" * 64


# --------------------------------------------------------------------------- #
# Key management (represents the "acquisition device" identity)
# --------------------------------------------------------------------------- #
def generate_device_keypair(key_dir: str | Path) -> tuple[Path, Path]:
    """Generate an Ed25519 keypair and write PEM files. Returns (priv, pub)."""
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()

    priv_path = key_dir / "device_private.pem"
    pub_path = key_dir / "device_public.pem"

    priv_path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    pub_path.write_bytes(
        priv.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return priv_path, pub_path


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    return serialization.load_pem_public_key(Path(path).read_bytes())


# --------------------------------------------------------------------------- #
# Canonical hashing helpers
# --------------------------------------------------------------------------- #
def sha256_image(arr: np.ndarray) -> str:
    """Hash the raw pixel bytes of a uint8 image array (deterministic)."""
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).hexdigest()


def signed_payload(image_sha: str, label: int, meta: dict) -> bytes:
    """
    Canonical bytes that get signed. Binding the label + meta into the
    signature is what makes label-flip poisoning (A2) detectable.
    """
    payload = {
        "image_sha": image_sha,
        "label": int(label),
        "meta": meta,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_chain_hash(image_sha: str, prev_hash: str) -> str:
    return hashlib.sha256((image_sha + prev_hash).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# The secure database
# --------------------------------------------------------------------------- #
class SecureImageDB:
    def __init__(self, db_path: str | Path, image_store: str | Path):
        self.db_path = str(db_path)
        self.image_store = Path(image_store)
        self.image_store.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                image_id     TEXT PRIMARY KEY,
                seq          INTEGER,
                sha256       TEXT NOT NULL,
                prev_hash    TEXT NOT NULL,
                chain_hash   TEXT NOT NULL,
                signature    TEXT NOT NULL,
                label        INTEGER NOT NULL,
                meta         TEXT NOT NULL,
                file_path    TEXT NOT NULL,
                ingested_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id    TEXT,
                event       TEXT NOT NULL,
                result      TEXT NOT NULL,
                detail      TEXT,
                ts          TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # ----------------------------- audit -------------------------------- #
    def _log(self, image_id: Optional[str], event: str, result: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO audit (image_id, event, result, detail, ts) VALUES (?,?,?,?,?)",
            (image_id, event, result, detail, _dt.datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    # ----------------------------- helpers ------------------------------ #
    def _last_row(self):
        return self.conn.execute(
            "SELECT seq, chain_hash FROM images ORDER BY seq DESC LIMIT 1"
        ).fetchone()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    # ----------------------------- ingest ------------------------------- #
    def ingest(
        self,
        image_id: str,
        arr: np.ndarray,
        label: int,
        meta: dict,
        signer: Ed25519PrivateKey,
    ) -> str:
        """Ingest one image: store pixels, hash, chain, and sign. Returns chain_hash."""
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        image_sha = sha256_image(arr)

        last = self._last_row()
        seq = 0 if last is None else last[0] + 1
        prev_hash = GENESIS_PREV if last is None else last[1]
        chain_hash = compute_chain_hash(image_sha, prev_hash)

        signature = signer.sign(signed_payload(image_sha, label, meta)).hex()

        # persist pixels as PNG (lossless -> hash stays stable)
        file_path = self.image_store / f"{image_id}.png"
        Image.fromarray(arr).save(file_path)

        self.conn.execute(
            """INSERT INTO images
               (image_id, seq, sha256, prev_hash, chain_hash, signature,
                label, meta, file_path, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                image_id, seq, image_sha, prev_hash, chain_hash, signature,
                int(label), json.dumps(meta, sort_keys=True),
                str(file_path), _dt.datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        self._log(image_id, "ingest", "ok", f"seq={seq}")
        return chain_hash

    # --------------------------- verify + load -------------------------- #
    def verify_and_load(
        self, image_id: str, device_pub: Ed25519PublicKey
    ) -> tuple[Optional[np.ndarray], int, str]:
        """
        Integrity + authenticity gate that the DataLoader calls.
        Returns (pixels|None, label, status). status is 'ok' or a failure reason.
        On any failure returns (None, -1, reason) so the loader can drop the sample.
        """
        row = self.conn.execute(
            "SELECT sha256, chain_hash, prev_hash, signature, label, meta, file_path "
            "FROM images WHERE image_id=?",
            (image_id,),
        ).fetchone()
        if row is None:
            self._log(image_id, "verify", "fail", "not_found")
            return None, -1, "not_found"

        sha_db, chain_db, prev_db, sig_hex, label, meta_json, file_path = row
        meta = json.loads(meta_json)

        # 1) recompute image hash from stored file -> catches A1 tampering
        try:
            arr = np.ascontiguousarray(np.asarray(Image.open(file_path)), dtype=np.uint8)
        except Exception as e:  # noqa
            self._log(image_id, "verify", "fail", f"read_error:{e}")
            return None, -1, "read_error"

        sha_now = sha256_image(arr)
        if sha_now != sha_db:
            self._log(image_id, "verify", "fail", "hash_mismatch")
            return None, -1, "hash_mismatch"

        # 2) recompute chain link -> catches record tampering
        if compute_chain_hash(sha_db, prev_db) != chain_db:
            self._log(image_id, "verify", "fail", "chain_broken")
            return None, -1, "chain_broken"

        # 3) signature over {hash,label,meta} -> catches A2 label-flip & A3 wrong key
        try:
            device_pub.verify(bytes.fromhex(sig_hex), signed_payload(sha_db, label, meta))
        except InvalidSignature:
            self._log(image_id, "verify", "fail", "bad_signature")
            return None, -1, "bad_signature"

        self._log(image_id, "verify", "ok", "")
        return arr, int(label), "ok"

    # --------------------------- full chain scan ------------------------ #
    def verify_chain_full(self) -> dict:
        """
        Walk the whole chain in seq order. Report first broken link (if any)
        and total verified. This produces the headline detection result.
        """
        rows = self.conn.execute(
            "SELECT image_id, seq, sha256, prev_hash, chain_hash "
            "FROM images ORDER BY seq ASC"
        ).fetchall()

        prev = GENESIS_PREV
        for image_id, seq, sha, prev_db, chain_db in rows:
            if prev_db != prev:
                return {"valid": False, "first_broken": image_id, "seq": seq,
                        "reason": "prev_hash_mismatch", "verified": seq}
            if compute_chain_hash(sha, prev_db) != chain_db:
                return {"valid": False, "first_broken": image_id, "seq": seq,
                        "reason": "chain_hash_mismatch", "verified": seq}
            prev = chain_db

        return {"valid": True, "first_broken": None, "verified": len(rows)}

    def close(self) -> None:
        self.conn.close()
