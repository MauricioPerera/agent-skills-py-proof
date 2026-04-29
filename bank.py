#!/usr/bin/env python3
"""agent-skills bank — minimal Python implementation, proof that spec v0.2
is sufficient for a second implementation.

What's in scope (per the spec the reference CLI also implements):
  - SKILL.md parse + validate (subset of SPEC §2.2 required fields)
  - sync (resolve ref → SHA via GitHub API, fetch skills-index.json + each
    SKILL.md from jsDelivr CDN, embed each, store)
  - query (embed intent, cosine over indexed skills, return top-K)
  - bench (run a JSONL/JSON-array truth file, report top-K accuracy)

What's deliberately out of scope (orthogonal to retrieval validation):
  - exec (subprocess + audit log)
  - rerank (intent-conditional / global) — pure cosine is enough for proof
  - signature verification
  - init / publish (author tooling)

Embedding provider: Ollama (local, zero credentials), per spec §4.7.
Default model: embeddinggemma (768-dim) — same model used in the
reference CLI's live BENCHMARK numbers.

Goal: run bench against agent-skills-pack@main + bench-truth.jsonl and
confirm top-K accuracy is within tolerance of the reference CLI's numbers
(34/35 top-1, 35/35 top-3 on Ollama embeddinggemma). If yes → spec v0.2
demonstrably supports an independent implementation.

Usage:
    python bank.py sync github.com/MauricioPerera/agent-skills-pack@main
    python bank.py query "fetch the contents of a URL"
    python bank.py bench bench-truth.jsonl

Bank state is at ~/.config/agent-skills-py/ (XDG-style), separate from the
TS CLI's bank to avoid interference.

Dependencies: pyyaml, requests. (Python ≥ 3.10.)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "embeddinggemma")
# Known dimensions; spec §4.7 requires the bank to record (name, dim) pairs.
KNOWN_DIMS: dict[str, int] = {
    "embeddinggemma": 768,
    "embeddinggemma:latest": 768,
    "nomic-embed-text": 768,
    "nomic-embed-text:latest": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "bge-m3": 1024,
}

REQUIRED_FIELDS = {"schema_version", "id", "version", "title", "description",
                   "use_when", "command_template"}


def bank_root() -> Path:
    """Default bank root, separate from the TS CLI's bank to avoid interference."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "agent-skills-py"


# ─────────────────────────────────────────────────────────────────────────
# SKILL.md parsing + validation (SPEC §2.1, §2.2)
# ─────────────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


def parse_skill_md(source: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into (frontmatter dict, body str). Per SPEC §2.1.

    Raises ValueError on malformed input.
    """
    m = _FRONTMATTER_RE.match(source)
    if not m:
        raise ValueError("SKILL.md is missing YAML frontmatter delimited by ---")
    fm_text, body = m.groups()
    try:
        fm = yaml.safe_load(fm_text)
    except yaml.YAMLError as e:
        raise ValueError(f"frontmatter YAML parse error: {e}") from None
    if not isinstance(fm, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return fm, body


def validate_skill(fm: dict[str, Any]) -> list[str]:
    """Return list of validation errors. Empty list = valid. Subset of SPEC §2.2."""
    errors: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in fm:
            errors.append(f"missing required field: {f}")
    if "id" in fm and not re.match(r"^[a-zA-Z0-9_-]+$", str(fm["id"])):
        errors.append("id must match ^[a-zA-Z0-9_-]+$")
    return errors


# ─────────────────────────────────────────────────────────────────────────
# Embedding provider: Ollama (SPEC §4.7 — name + dim + embed triplet)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Embedder:
    name: str
    dim: int
    base_url: str

    @classmethod
    def ollama(cls, base_url: str = OLLAMA_BASE_URL,
               model: str = OLLAMA_MODEL) -> "Embedder":
        dim = KNOWN_DIMS.get(model)
        if dim is None:
            # Probe by sending one embed call; cheaper than asking the user.
            r = requests.post(
                f"{base_url.rstrip('/')}/api/embed",
                json={"model": model, "input": "probe"},
                timeout=30,
            )
            r.raise_for_status()
            dim = len(r.json()["embeddings"][0])
        return cls(name=f"ollama:{model}", dim=dim, base_url=base_url.rstrip("/"))

    def embed(self, text: str) -> list[float]:
        if not text:
            raise ValueError("cannot embed empty text")
        # Extract model name from "ollama:<model>"
        model = self.name.split(":", 1)[1] if ":" in self.name else self.name
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=60,
        )
        if not r.ok:
            raise RuntimeError(f"Ollama returned {r.status_code}: {r.text[:200]}")
        vec = r.json()["embeddings"][0]
        if len(vec) != self.dim:
            raise RuntimeError(
                f"Ollama returned {len(vec)}-dim vector; expected {self.dim}",
            )
        return vec


def cosine(a: list[float], b: list[float]) -> float:
    """SPEC-compliant cosine similarity over equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def compose_embedding_text(fm: dict[str, Any]) -> str:
    """Per SPEC §4.2: title . use_when . description . examples[].intent . tags

    Joined by '. ' (period + space). The reference CLI uses this exact
    composition; aligning here is what makes vectors comparable across
    implementations of the same model.
    """
    parts: list[str] = [
        str(fm.get("title", "")),
        str(fm.get("use_when", "")),
        str(fm.get("description", "")),
    ]
    examples = fm.get("examples")
    if isinstance(examples, list) and examples:
        intents = [str(e.get("intent", "")) for e in examples
                   if isinstance(e, dict)]
        if intents:
            parts.append("\n".join(intents))
    tags = fm.get("tags")
    if isinstance(tags, list) and tags:
        parts.append(" ".join(str(t) for t in tags))
    return ". ".join(parts)


# ─────────────────────────────────────────────────────────────────────────
# Signature verification (SPEC §5.1 Level 3a — host-verified)
# ─────────────────────────────────────────────────────────────────────────
#
# v0.2 of the spec splits Level 3 (signed tags) into 3a (host-verified —
# the bank trusts the host's GPG verification API) and 3b (client-verified —
# the bank verifies locally against `trusted_keys`). This proof implements
# 3a only, mirroring the reference TS CLI v0.10.0. 3b would require a local
# GPG keyring and is properly the next iteration.
#
# The verifier returns one of four statuses, exactly per SPEC §5.1:
#   "valid"      — host returned verified=true
#   "invalid"    — verified=false but a signature is present (unknown_key,
#                  bad_email, expired, etc.) — active red flag
#   "unsigned"   — verified=false and no signature payload (annotated tag
#                  but no GPG signature, OR lightweight tag with no tag
#                  object) — sloppy publisher hygiene, not malicious
#   "unverified" — bank could not perform verification (non-GitHub host,
#                  raw SHA ref, API rate limit, etc.) — caller decides
#                  whether to treat as failure or as "no signal"


def detect_signature_method(signature: str | None) -> str | None:
    """Mirror of TS detectSignatureMethod (v0.14.0+, "ssh" added v0.15.0).

    Structural detection of the signing method by PEM header:
      "-----BEGIN PGP SIGNATURE-----"  -> "gpg"
      "-----BEGIN SSH SIGNATURE-----"  -> "ssh"      (git's gpg.format=ssh)
      "-----BEGIN SIGNED MESSAGE-----" -> "sigstore" (gitsign / Fulcio)

    Returns None for unrecognised / missing payloads. This is detection,
    NOT verification - the trust verdict still comes from the host's
    'verified' field.

    For Sigstore, see SPEC v0.3.2 sigstore-on-host trap: a 'bad_cert'
    verdict on a 'sigstore'-method tag is ambiguous (Fulcio cert may have
    expired post-sign), not equivalent to a forged signature.
    """
    if not signature:
        return None
    if "-----BEGIN PGP SIGNATURE-----" in signature:
        return "gpg"
    if "-----BEGIN SSH SIGNATURE-----" in signature:
        return "ssh"
    if "-----BEGIN SIGNED MESSAGE-----" in signature:
        return "sigstore"
    return None


# ────────────────────────────────────────────────────────────────────
# Sigstore identity extraction (parity with TS CLI cms.ts, v0.16.0+)
# ────────────────────────────────────────────────────────────────────
# A "sigstore"-method tag carries a CMS SignedData blob whose first cert is
# the Fulcio-issued ephemeral signing cert. Its SAN carries the OIDC subject
# (email or workflow URI) and Fulcio extension OID 1.3.6.1.4.1.57264.1.1
# (or .1.8) carries the OIDC issuer. Together they answer "who signed this?".
# We hand-roll just enough ASN.1 to walk to those fields — no new deps.
# Cross-impl parity with the TS CLI is validated continuously via the e2e
# workflow (compares .signature.identity between implementations).

_PEM_CMS_OPEN = "-----BEGIN SIGNED MESSAGE-----"
_PEM_CMS_CLOSE = "-----END SIGNED MESSAGE-----"
_FULCIO_OIDC_ISSUER_OID_V1 = "1.3.6.1.4.1.57264.1.1"
_FULCIO_OIDC_ISSUER_OID_V2 = "1.3.6.1.4.1.57264.1.8"
_X509_SAN_EXT_OID = "2.5.29.17"


def _read_tlv(buf: bytes, off: int) -> tuple[int, int, int, int]:
    """Read one DER TLV. Returns (tag, value_off, value_len, total_len)."""
    if off + 2 > len(buf):
        raise ValueError("ASN.1 truncated at TLV header")
    tag = buf[off]
    p = off + 1
    length = buf[p]
    p += 1
    if length & 0x80:
        n = length & 0x7F
        if n == 0:
            raise ValueError("ASN.1 indefinite length not supported")
        if n > 4:
            raise ValueError(f"ASN.1 length-of-length {n} unreasonable")
        if p + n > len(buf):
            raise ValueError("ASN.1 truncated in length bytes")
        length = 0
        for _ in range(n):
            length = (length << 8) | buf[p]
            p += 1
    if p + length > len(buf):
        raise ValueError("ASN.1 declared length exceeds buffer")
    return tag, p, length, (p - off + length)


def _decode_oid(buf: bytes) -> str:
    if not buf:
        return ""
    out = [str(buf[0] // 40), str(buf[0] % 40)]
    v = 0
    for b in buf[1:]:
        v = (v << 7) | (b & 0x7F)
        if not (b & 0x80):
            out.append(str(v))
            v = 0
    return ".".join(out)


def _extract_first_cert_der(pem_signature: str) -> bytes | None:
    """Walk CMS SignedData to extract the first cert's DER bytes."""
    start = pem_signature.find(_PEM_CMS_OPEN)
    if start < 0:
        return None
    end = pem_signature.find(_PEM_CMS_CLOSE, start + len(_PEM_CMS_OPEN))
    if end < 0:
        return None
    b64 = re.sub(r"\s+", "", pem_signature[start + len(_PEM_CMS_OPEN):end])
    try:
        der = base64.b64decode(b64, validate=True)
    except Exception:
        return None
    if not der:
        return None
    try:
        # ContentInfo SEQUENCE -> skip OID -> [0] EXPLICIT -> SignedData
        ci_tag, ci_voff, ci_vlen, _ = _read_tlv(der, 0)
        if ci_tag != 0x30:
            return None
        off = ci_voff
        _, _, _, oid_total = _read_tlv(der, off)
        off += oid_total
        ex_tag, ex_voff, _, _ = _read_tlv(der, off)
        if ex_tag != 0xA0:
            return None
        sd_tag, sd_voff, sd_vlen, _ = _read_tlv(der, ex_voff)
        if sd_tag != 0x30:
            return None
        # Walk SignedData children for [0] IMPLICIT certificates (tag 0xa0).
        p = sd_voff
        sd_end = sd_voff + sd_vlen
        cert_set = None
        while p < sd_end:
            t_tag, t_voff, t_vlen, t_total = _read_tlv(der, p)
            if t_tag == 0xA0:
                cert_set = (t_voff, t_vlen)
                break
            p += t_total
        if cert_set is None:
            return None
        # First cert SEQUENCE inside the set.
        c_tag, _, _, c_total = _read_tlv(der, cert_set[0])
        if c_tag != 0x30:
            return None
        return der[cert_set[0]:cert_set[0] + c_total]
    except Exception:
        return None


def _walk_cert_extensions(cert_der: bytes):
    """Yield (oid_str, octet_string_value_bytes) for each extension."""
    cert_tag, cert_voff, _, _ = _read_tlv(cert_der, 0)
    tbs_tag, tbs_voff, tbs_vlen, _ = _read_tlv(cert_der, cert_voff)
    p = tbs_voff
    tbs_end = tbs_voff + tbs_vlen
    ext_outer = None
    while p < tbs_end:
        t_tag, t_voff, t_vlen, t_total = _read_tlv(cert_der, p)
        if t_tag == 0xA3:  # [3] EXPLICIT extensions
            ext_outer = (t_voff, t_vlen)
            break
        p += t_total
    if ext_outer is None:
        return
    seq_tag, seq_voff, seq_vlen, _ = _read_tlv(cert_der, ext_outer[0])
    if seq_tag != 0x30:
        return
    ep = seq_voff
    e_end = seq_voff + seq_vlen
    while ep < e_end:
        ext_tag, ext_voff, _, ext_total = _read_tlv(cert_der, ep)
        ip = ext_voff
        oid_tag, oid_voff, oid_vlen, oid_total = _read_tlv(cert_der, ip)
        oid = _decode_oid(cert_der[oid_voff:oid_voff + oid_vlen])
        ip += oid_total
        v_tag, v_voff, v_vlen, v_total = _read_tlv(cert_der, ip)
        if v_tag == 0x01:  # optional critical BOOLEAN
            ip += v_total
            v_tag, v_voff, v_vlen, _ = _read_tlv(cert_der, ip)
        if v_tag == 0x04:  # OCTET STRING
            yield oid, cert_der[v_voff:v_voff + v_vlen]
        ep += ext_total


def _parse_san(san_octet_value: bytes) -> tuple[str, str] | None:
    """Parse the SAN extension's OCTET STRING contents.
    Returns (subject, subject_type) for the first entry."""
    try:
        seq_tag, seq_voff, seq_vlen, _ = _read_tlv(san_octet_value, 0)
        if seq_tag != 0x30:
            return None
        # First GeneralName entry — context-specific tag (0x80 | choice).
        gn_tag, gn_voff, gn_vlen, _ = _read_tlv(san_octet_value, seq_voff)
        value = san_octet_value[gn_voff:gn_voff + gn_vlen].decode("utf-8", errors="replace")
        if gn_tag == 0x81:  # rfc822Name
            return value, "email"
        if gn_tag == 0x86:  # uniformResourceIdentifier
            return value, "uri"
        return value, "other"
    except Exception:
        return None


def extract_sigstore_identity(pem_signature: str | None) -> dict[str, str] | None:
    """Top-level: extract Sigstore identity claim from a CMS payload.

    Returns {"subject", "subject_type", "issuer"?} or None on missing /
    malformed input. Mirrors TS extractSigstoreIdentity (cms.ts).

    IMPORTANT: extraction != verification. The returned identity is the
    cert's *claimed* identity; verifying against Rekor is Level 4 work.
    """
    if not pem_signature:
        return None
    cert_der = _extract_first_cert_der(pem_signature)
    if not cert_der:
        return None

    subject = subject_type = issuer = None
    for oid, value in _walk_cert_extensions(cert_der):
        if oid == _X509_SAN_EXT_OID and subject is None:
            parsed = _parse_san(value)
            if parsed:
                subject, subject_type = parsed
        elif oid in (_FULCIO_OIDC_ISSUER_OID_V1, _FULCIO_OIDC_ISSUER_OID_V2) and issuer is None:
            # V1: bare UTF-8 in the OCTET STRING. V2: UTF-8String DER (tag 0x0c).
            if oid == _FULCIO_OIDC_ISSUER_OID_V2 and len(value) > 0 and value[0] == 0x0C:
                try:
                    _, voff, vlen, _ = _read_tlv(value, 0)
                    issuer = value[voff:voff + vlen].decode("utf-8", errors="replace")
                except Exception:
                    issuer = value.decode("utf-8", errors="replace")
            else:
                issuer = value.decode("utf-8", errors="replace")

    if subject is None:
        return None
    out: dict[str, str] = {"subject": subject, "subject_type": subject_type or "other"}
    if issuer is not None:
        out["issuer"] = issuer
    return out


# ────────────────────────────────────────────────────────────────────
# gitsign Rekor lookup hash — parity with TS CLI v0.17.1 (cms.ts)
# ────────────────────────────────────────────────────────────────────
# gitsign submits Rekor entries indexed by SHA-256 of the SignerInfo's
# SignedAttrs "marshaled for verification" (RFC 5652 §5.4). The [0]
# IMPLICIT signedAttrs are re-encoded with an explicit SET tag (0x31)
# instead of the implicit context-specific [0] (0xa0) — same length,
# same content, different outer tag byte. SHA-256 of those bytes is the
# hash a Level 4 verifier uses to locate the corresponding Rekor entry
# via /api/v1/index/retrieve. See SPEC §5.4.2 step 3.
#
# Validated structurally via the messageDigest invariant (RFC 5652 §11.2):
# the messageDigest attribute INSIDE SignedAttrs equals SHA-256(payload).
# Tests confirm this against the real sigstore/gitsign@v0.14.0 fixture
# in cms.test.ts — Python output is byte-identical to TS by design.

_PEM_CMS_OPEN_GITSIGN = "-----BEGIN SIGNED MESSAGE-----"
_PEM_CMS_CLOSE_GITSIGN = "-----END SIGNED MESSAGE-----"
_REKOR_PUBLIC_HOST = "https://rekor.sigstore.dev"


def _extract_signed_attrs_bytes(pem_signature: str) -> bytes | None:
    """Walk a CMS payload to its first SignerInfo's SignedAttrs and
    return the raw inner bytes (the Attribute SEQUENCEs).

    Mirrors TS extractSignedAttrsBytes (cms.ts internal helper). Returns
    None on any malformed structure — callers treat extraction failure
    as "not a recognisable Sigstore CMS payload".
    """
    start = pem_signature.find(_PEM_CMS_OPEN_GITSIGN)
    if start < 0:
        return None
    end = pem_signature.find(_PEM_CMS_CLOSE_GITSIGN, start + len(_PEM_CMS_OPEN_GITSIGN))
    if end < 0:
        return None
    b64 = re.sub(r"\s+", "", pem_signature[start + len(_PEM_CMS_OPEN_GITSIGN):end])
    try:
        der = base64.b64decode(b64, validate=True)
    except Exception:
        return None
    if not der:
        return None

    try:
        # ContentInfo SEQUENCE -> skip OID -> [0] EXPLICIT -> SignedData SEQUENCE
        ci_tag, ci_voff, _, _ = _read_tlv(der, 0)
        if ci_tag != 0x30:
            return None
        off = ci_voff
        _, _, _, oid_total = _read_tlv(der, off)
        off += oid_total
        ex_tag, ex_voff, _, _ = _read_tlv(der, off)
        if ex_tag != 0xA0:
            return None
        sd_tag, sd_voff, sd_vlen, _ = _read_tlv(der, ex_voff)
        if sd_tag != 0x30:
            return None

        # Walk SignedData children. SignerInfos is the LAST SET (tag 0x31)
        # — the digestAlgorithms SET comes earlier. Take the last one.
        p = sd_voff
        sd_end = sd_voff + sd_vlen
        signer_infos = None
        while p < sd_end:
            t_tag, t_voff, t_vlen, t_total = _read_tlv(der, p)
            if t_tag == 0x31:
                signer_infos = (t_voff, t_vlen)
            p += t_total
        if signer_infos is None:
            return None

        # First SignerInfo SEQUENCE inside the set.
        si_tag, si_voff, si_vlen, _ = _read_tlv(der, signer_infos[0])
        if si_tag != 0x30:
            return None

        # Walk SignerInfo children to find signedAttrs ([0] IMPLICIT, tag 0xa0).
        # Order per RFC 5652 §5.3: version, sid, digestAlgorithm, signedAttrs?, ...
        sip = si_voff
        si_end = si_voff + si_vlen
        while sip < si_end:
            t_tag, t_voff, t_vlen, t_total = _read_tlv(der, sip)
            if t_tag == 0xA0:
                return der[t_voff:t_voff + t_vlen]
            sip += t_total
        return None
    except Exception:
        return None


def _encode_der_length(n: int) -> bytes:
    """DER-encode a length value per X.690 §8.1.3."""
    if n < 128:
        return bytes([n])
    if n < 256:
        return bytes([0x81, n])
    if n < 65536:
        return bytes([0x82, (n >> 8) & 0xFF, n & 0xFF])
    if n < 16777216:
        return bytes([0x83, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF])
    return bytes([
        0x84,
        (n >> 24) & 0xFF,
        (n >> 16) & 0xFF,
        (n >> 8) & 0xFF,
        n & 0xFF,
    ])


def compute_gitsign_rekor_lookup_hash(pem_signature: str | None) -> str | None:
    """Compute the gitsign-flavor Rekor lookup hash for a CMS payload.

    Returns lower-case hex SHA-256 of the SignerInfo's SignedAttrs
    marshaled for verification (the [0]-tagged signedAttrs re-encoded
    with the SET tag 0x31, length and content unchanged).

    Use this hash to locate the corresponding Rekor entry via
    `find_rekor_entry_by_hash` (or any other Rekor index/retrieve client).

    Returns None if the input isn't a parseable Sigstore CMS payload or
    the SignedAttrs aren't present. Cross-impl parity with TS
    computeGitsignRekorLookupHash (cms.ts) — produces byte-identical
    output on the same input.
    """
    if not pem_signature:
        return None
    inner = _extract_signed_attrs_bytes(pem_signature)
    if inner is None:
        return None
    # Re-frame: SET tag (0x31) + DER-length + content bytes (unchanged).
    reframed = bytes([0x31]) + _encode_der_length(len(inner)) + inner
    return hashlib.sha256(reframed).hexdigest()


def find_rekor_entry_by_hash(
    hash_hex: str,
    host: str = _REKOR_PUBLIC_HOST,
    timeout: int = 30,
) -> list[str]:
    """Look up Rekor entry UUIDs by an artifact hash via
    POST /api/v1/index/retrieve.

    Returns a list of UUIDs (may be empty — no entry matches is NOT an
    error; old or rotated Rekor shards may have pruned entries even
    when the signing event was real).

    `hash_hex` must be a 64-char lower-case hex SHA-256 (the function
    does not accept a sha256: prefix; it adds one before sending).
    Mirrors TS findRekorEntryByHash (rekor.ts).
    """
    if not re.match(r"^[a-f0-9]{64}$", hash_hex):
        raise ValueError(
            f"rekor: hash must be 64-char lower-case hex SHA-256, got '{hash_hex}'",
        )
    url = f"{host}/api/v1/index/retrieve"
    r = requests.post(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={"hash": f"sha256:{hash_hex}"},
        timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(
            f"rekor: index/retrieve returned {r.status_code} {r.reason}",
        )
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError("rekor: index/retrieve did not return an array")
    for u in data:
        if not isinstance(u, str):
            raise RuntimeError("rekor: index/retrieve array contains non-string")
    return data


def verify_github_tag(repo: str, ref: str) -> dict[str, Any]:
    """Mirror SPEC §5.1 Level 3a verification via GitHub's API.

    Returns: {"status": <SignatureStatus>, "reason": str, "signed_by"?: str,
              "method"?: "gpg" | "ssh" | "sigstore",
              "identity"?: {"subject", "subject_type", "issuer"?}}
    The "method" field (v0.14.0+) is structural detection of which crypto
    system signed the tag. The "identity" field (v0.16.0+) is extracted
    from the CMS for sigstore-method tags. Full Sigstore Level 4 (Rekor
    inclusion-proof verification) remains queued.
    """
    if not repo.startswith("github.com/"):
        return {"status": "unverified",
                "reason": f"host '{repo}' not supported by GPG verifier yet"}
    if re.match(r"^[a-f0-9]{40,}$", ref):
        return {"status": "unverified",
                "reason": "ref is a raw commit hash; no tag-level signature to verify"}

    owner_repo = repo[len("github.com/"):]
    headers = {"Accept": "application/vnd.github+json"}

    # 1. Resolve tag → tag-object SHA (annotated tag) or commit SHA (lightweight)
    ref_url = f"https://api.github.com/repos/{owner_repo}/git/refs/tags/{urllib.parse.quote(ref)}"
    r = requests.get(ref_url, headers=headers, timeout=30)
    if r.status_code == 404:
        return {"status": "unverified", "reason": f"tag '{ref}' not found via GitHub API"}
    if not r.ok:
        return {"status": "unverified",
                "reason": f"GitHub API returned {r.status_code} {r.reason}"}
    obj = r.json().get("object") or {}
    if obj.get("type") == "commit":
        return {"status": "unsigned",
                "reason": "lightweight tag (annotated tag required for tag-signing)"}
    if obj.get("type") != "tag" or not isinstance(obj.get("sha"), str):
        return {"status": "unverified", "reason": "unexpected ref shape (no tag object)"}

    # 2. Fetch tag object + verification block
    tag_url = f"https://api.github.com/repos/{owner_repo}/git/tags/{obj['sha']}"
    r = requests.get(tag_url, headers=headers, timeout=30)
    if not r.ok:
        return {"status": "unverified",
                "reason": f"tag-object fetch returned {r.status_code} {r.reason}"}
    tag = r.json()
    tagger = tag.get("tagger") or {}
    name = tagger.get("name") or ""
    email = tagger.get("email") or ""
    signed_by = (f"{name} <{email}>" if name and email
                 else (f"<{email}>" if email else (name or None)))
    out: dict[str, Any] = {"signed_by": signed_by} if signed_by else {}

    verification = tag.get("verification")
    if not isinstance(verification, dict):
        out.update({"status": "unverified", "reason": "GitHub returned no verification block"})
        return out

    reason = verification.get("reason", "unknown")
    method = detect_signature_method(verification.get("signature"))
    # For sigstore-method tags, attempt CMS identity extraction (v0.16.0+).
    identity = (extract_sigstore_identity(verification.get("signature"))
                if method == "sigstore" else None)

    if verification.get("verified") is True:
        out.update({"status": "valid", "reason": reason})
        if method is not None:
            out["method"] = method
        if identity is not None:
            out["identity"] = identity
        return out

    # verified=false. Distinguish "no signature attempted" from "signature present
    # but couldn't be verified" — operators care about the difference.
    if reason == "unsigned" or verification.get("signature") in (None, ""):
        out.update({"status": "unsigned", "reason": reason})
        return out
    out.update({"status": "invalid", "reason": reason})
    if method is not None:
        out["method"] = method
    if identity is not None:
        out["identity"] = identity
    return out


def enforce_verification(result: dict[str, Any], repo: str, ref: str) -> None:
    """Raise RuntimeError if status != 'valid'. Used when --verify-signature is set."""
    if result.get("status") == "valid":
        return
    detail = (f" (tagger: {result['signed_by']}, reason: {result['reason']})"
              if result.get("signed_by") else f" (reason: {result.get('reason', 'unknown')})")
    raise RuntimeError(
        f"signature verification failed for {repo}@{ref}: status={result.get('status')}{detail}. "
        f"Pass without --verify-signature to ingest unverified, or work with the publisher to "
        f"sign their tag with 'git tag -s' and re-tag.",
    )


# ─────────────────────────────────────────────────────────────────────────
# Sync (SPEC §7 — resolve ref → SHA → fetch index → fetch skills → embed)
# ─────────────────────────────────────────────────────────────────────────


def parse_source_spec(source: str) -> tuple[str, str]:
    """`<host>/<owner>/<repo>[@<ref>]` → (repo, ref). Default ref = main."""
    if "@" in source:
        repo, ref = source.split("@", 1)
        return repo, ref
    return source, "main"


def resolve_ref(repo: str, ref: str) -> str:
    """Tag/branch/commit → 40+hex SHA via GitHub API. Per SPEC §7.1."""
    if re.match(r"^[a-f0-9]{40,}$", ref):
        return ref
    if not repo.startswith("github.com/"):
        raise ValueError(
            f"non-GitHub host '{repo}' not supported by this proof-of-concept",
        )
    owner_repo = repo[len("github.com/"):]
    candidates = [
        f"https://api.github.com/repos/{owner_repo}/git/refs/tags/{urllib.parse.quote(ref)}",
        f"https://api.github.com/repos/{owner_repo}/commits/{urllib.parse.quote(ref)}",
    ]
    headers = {"Accept": "application/vnd.github+json"}
    for url in candidates:
        r = requests.get(url, headers=headers, timeout=30)
        if not r.ok:
            continue
        data = r.json()
        sha = data.get("object", {}).get("sha") if isinstance(data.get("object"), dict) else data.get("sha")
        if isinstance(sha, str) and re.match(r"^[a-f0-9]{40,}$", sha):
            return sha
    raise RuntimeError(f"cannot resolve ref '{ref}' for {repo}")


def cdn_url(repo: str, sha: str, path: str) -> str:
    if not repo.startswith("github.com/"):
        raise ValueError(f"unsupported host: {repo}")
    owner_repo = repo[len("github.com/"):]
    return f"https://cdn.jsdelivr.net/gh/{owner_repo}@{sha}/{path}"


@dataclass
class IndexedSkill:
    identity: str
    short_id: str
    title: str
    use_when: str
    description: str
    embedding: list[float]
    embedding_model: str
    skill_md: str = ""  # raw source for round-trip


def cmd_sync(source: str, embedder: Embedder, root: Path,
             verify_signature: bool = False) -> dict[str, Any]:
    repo, ref_requested = parse_source_spec(source)
    sha = resolve_ref(repo, ref_requested)

    # SPEC §5.1: always observe; optionally enforce. Status is recorded in
    # provenance regardless. With verify_signature=True, anything other
    # than "valid" aborts BEFORE any embedding API call (operators don't
    # pay for a sync that we're going to refuse to ingest).
    signature = verify_github_tag(repo, ref_requested)
    if verify_signature:
        enforce_verification(signature, repo, ref_requested)

    index_url = cdn_url(repo, sha, "skills-index.json")
    r = requests.get(index_url, timeout=30)
    r.raise_for_status()
    index = r.json()
    skills_idx = index.get("skills") or []
    if not isinstance(skills_idx, list):
        raise RuntimeError(f"skills-index.json missing 'skills' array")

    # Initialize / verify bank metadata (SPEC §4.7 — refuse model mix).
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedding_model") != embedder.name:
            raise RuntimeError(
                f"bank already initialized with '{meta['embedding_model']}'; "
                f"refusing to mix with '{embedder.name}'. Delete {root} to start over.",
            )
    else:
        meta_path.write_text(
            json.dumps({
                "schema_version": "0.1",
                "embedding_model": embedder.name,
                "embedding_dim": embedder.dim,
            }, indent=2),
            encoding="utf-8",
        )

    skills_dir = root / "skills"
    skills_dir.mkdir(exist_ok=True)
    results = []

    for entry in skills_idx:
        sid = entry["id"]
        # Per SPEC §3.2: skill URL via entry.url OR url_template.
        url = entry.get("url")
        if not url:
            tpl = index.get("url_template")
            if not tpl:
                results.append({"id": sid, "status": "error",
                                "message": "no url + no url_template"})
                continue
            url = tpl.replace("{ref}", sha).replace("{path}", sid)

        try:
            sr = requests.get(url, timeout=30)
            sr.raise_for_status()
            src = sr.text
            fm, _body = parse_skill_md(src)
            errs = validate_skill(fm)
            if errs:
                results.append({"id": sid, "status": "invalid", "errors": errs})
                continue
        except Exception as e:
            results.append({"id": sid, "status": "error", "message": str(e)})
            continue

        # Embed per SPEC §4.2 composition
        emb_text = compose_embedding_text(fm)
        try:
            vec = embedder.embed(emb_text)
        except Exception as e:
            results.append({"id": sid, "status": "error",
                            "message": f"embedding failed: {e}"})
            continue

        identity = f"{repo}@{sha}/{sid}"
        # Provenance per SPEC §2.5 — populated at ingest time, not author-declared.
        provenance = {
            "source_type": "git",
            "source": repo,
            "ref_resolved_to": sha,
            "ref_requested": ref_requested,
            "fetched_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "signature_status": signature["status"],
        }
        if signature.get("signed_by"):
            provenance["signed_by"] = signature["signed_by"]
        if signature.get("method"):
            provenance["signature_method"] = signature["method"]
        record = {
            "identity": identity,
            "short_id": sid,
            "title": fm["title"],
            "use_when": fm["use_when"],
            "description": fm["description"],
            "embedding": vec,
            "embedding_model": embedder.name,
            "frontmatter": fm,
            "provenance": provenance,
        }
        # Filename: hash of identity (matches the TS CLI's behaviour conceptually,
        # but we don't need byte compatibility — only the spec-level contract).
        import hashlib
        fn = hashlib.sha256(identity.encode()).hexdigest()[:16] + ".json"
        (skills_dir / fn).write_text(
            json.dumps(record, indent=2), encoding="utf-8",
        )
        results.append({"id": sid, "identity": identity, "status": "synced"})

    summary = {
        "source": source,
        "ref_requested": ref_requested,
        "ref_resolved": sha,
        "total": len(results),
        "synced": sum(1 for r in results if r["status"] == "synced"),
        "invalid": sum(1 for r in results if r["status"] == "invalid"),
        "errored": sum(1 for r in results if r["status"] == "error"),
        "skills": results,
        "signature": signature,
        "signature_enforced": verify_signature and signature.get("status") == "valid",
    }

    # Persist subscription record so update / re-sync would work
    subs_path = root / "subscriptions.json"
    subs = json.loads(subs_path.read_text()) if subs_path.exists() else []
    subs = [s for s in subs if s.get("id") != source]
    sub: dict[str, Any] = {
        "id": source,
        "repo": repo,
        "ref_requested": ref_requested,
        "ref_resolved": sha,
    }
    if verify_signature:
        sub["verify_signature"] = True
    subs.append(sub)
    subs_path.write_text(json.dumps(subs, indent=2), encoding="utf-8")

    return summary


def list_skills(root: Path) -> list[IndexedSkill]:
    skills_dir = root / "skills"
    if not skills_dir.exists():
        return []
    out: list[IndexedSkill] = []
    for p in sorted(skills_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append(IndexedSkill(
                identity=d["identity"],
                short_id=d["short_id"],
                title=d["title"],
                use_when=d["use_when"],
                description=d["description"],
                embedding=d["embedding"],
                embedding_model=d["embedding_model"],
            ))
        except (OSError, json.JSONDecodeError, KeyError):
            # Per-entry corruption is non-fatal (SPEC §4.5 partial-failure
            # resilience). We catch ONLY the three classes of legitimate
            # corruption: filesystem read failures, malformed JSON, and
            # missing required fields. KeyboardInterrupt / MemoryError /
            # other system signals propagate so the operator can stop the
            # process cleanly.
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────
# Query (SPEC §4.3 — pure cosine, no rerank, no filter)
# ─────────────────────────────────────────────────────────────────────────


def cmd_query(intent: str, k: int, embedder: Embedder, root: Path) -> dict[str, Any]:
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    if meta["embedding_model"] != embedder.name:
        raise RuntimeError(
            f"embedding model mismatch: bank uses '{meta['embedding_model']}', "
            f"query is using '{embedder.name}'",
        )
    qvec = embedder.embed(intent)
    skills = list_skills(root)
    scored = [(cosine(qvec, s.embedding), s) for s in skills]
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:k]
    return {
        "intent": intent,
        "embedding_model": embedder.name,
        "hits": [
            {
                "identity": s.identity,
                "short_id": s.short_id,
                "title": s.title,
                "use_when": s.use_when,
                "score": round(score, 4),
            }
            for score, s in top
        ],
    }


# ─────────────────────────────────────────────────────────────────────────
# Bench (SPEC §4.6 — JSONL or JSON-array truth file)
# ─────────────────────────────────────────────────────────────────────────


def parse_truth_file(text: str, path: str) -> list[dict[str, str]]:
    """Auto-detect JSONL vs JSON-array per SPEC §4.6."""
    stripped = text.lstrip()
    if not stripped:
        raise ValueError(f"{path}: empty truth file")
    raw: list[Any]
    if stripped[0] == "[":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON: {e}") from None
        if not isinstance(raw, list):
            raise ValueError(f"{path}: top-level JSON value is not an array")
    else:
        raw = []
        for i, line in enumerate(text.split("\n"), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e}") from None

    out: list[dict[str, str]] = []
    for i, e in enumerate(raw, start=1):
        if not isinstance(e, dict):
            raise ValueError(f"{path} entry #{i}: expected object")
        intent = e.get("intent")
        expected = e.get("expected")
        if not isinstance(intent, str) or not intent:
            raise ValueError(f"{path} entry #{i}: missing/empty 'intent'")
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"{path} entry #{i}: missing/empty 'expected'")
        out.append({"intent": intent, "expected": expected})
    return out


def cmd_bench(truth_file: str, k: int, embedder: Embedder, root: Path) -> dict[str, Any]:
    text = Path(truth_file).read_text(encoding="utf-8")
    entries = parse_truth_file(text, truth_file)

    skills = list_skills(root)
    by_short_id: dict[str, list[IndexedSkill]] = {}
    for s in skills:
        by_short_id.setdefault(s.short_id, []).append(s)

    # Validate truth file: every expected MUST resolve to exactly one skill
    # (SPEC §4.6 — fail-fast contract).
    for e in entries:
        ms = by_short_id.get(e["expected"], [])
        if not ms:
            raise RuntimeError(f"truth file references unknown skill: {e['expected']}")
        if len(ms) > 1:
            raise RuntimeError(f"ambiguous expected '{e['expected']}': {len(ms)} matches")

    queries: list[dict[str, Any]] = []
    top1 = top3 = topk = 0
    total_top1_score = total_margin = 0.0

    for e in entries:
        expected_identity = by_short_id[e["expected"]][0].identity
        qvec = embedder.embed(e["intent"])
        scored = [(cosine(qvec, s.embedding), s) for s in skills]
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:max(k + 1, 10)]

        rank = None
        expected_score = None
        for i, (score, s) in enumerate(top[:k], start=1):
            if s.identity == expected_identity:
                rank = i
                expected_score = score
                break
        top1_score = top[0][0]
        margin = top1_score - (top[1][0] if len(top) > 1 else 0.0)

        if rank == 1:
            top1 += 1
        if rank is not None and rank <= 3:
            top3 += 1
        if rank is not None:
            topk += 1

        total_top1_score += top1_score
        total_margin += margin

        queries.append({
            "intent": e["intent"],
            "expected": e["expected"],
            "rank": rank,
            "got_top1": top[0][1].short_id,
            "top1_score": round(top1_score, 4),
            "expected_score": None if expected_score is None else round(expected_score, 4),
            "margin": round(margin, 4),
        })

    return {
        "truth_file": truth_file,
        "embedding_model": embedder.name,
        "total": len(entries),
        "top1": top1,
        "top3": top3,
        "topK": topk,
        "k": k,
        "mean_top1_score": round(total_top1_score / max(len(entries), 1), 4),
        "mean_margin": round(total_margin / max(len(entries), 1), 4),
        "queries": queries,
        "failures": [q for q in queries if q["rank"] != 1],
    }


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bank.py",
        description="agent-skills bank — minimal Python implementation (proof of spec v0.2)",
    )
    parser.add_argument("--bank-dir", help="override bank state directory")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="fetch + embed + index a pack")
    p_sync.add_argument("source", help="<host>/<owner>/<repo>[@<ref>]")
    p_sync.add_argument("--verify-signature", action="store_true",
                        help="abort if the resolved tag isn't GPG-verified by the host (SPEC §5.1 Level 3a)")

    p_query = sub.add_parser("query", help="embed intent + cosine search")
    p_query.add_argument("intent")
    p_query.add_argument("--k", type=int, default=5)

    p_bench = sub.add_parser("bench", help="run JSONL/JSON truth file")
    p_bench.add_argument("truth_file")
    p_bench.add_argument("--k", type=int, default=5)

    args = parser.parse_args()
    root = Path(args.bank_dir) if args.bank_dir else bank_root()

    embedder = Embedder.ollama()

    if args.cmd == "sync":
        result = cmd_sync(args.source, embedder, root,
                          verify_signature=args.verify_signature)
    elif args.cmd == "query":
        result = cmd_query(args.intent, args.k, embedder, root)
    elif args.cmd == "bench":
        result = cmd_bench(args.truth_file, args.k, embedder, root)
    else:
        parser.print_help()
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(args.cmd, result)
    if args.cmd == "bench" and result.get("failures"):
        return 1
    return 0


def _print_human(cmd: str, r: dict[str, Any]) -> None:
    """ASCII output to avoid console-encoding issues on Windows cp1252.
    For Unicode glyphs in your terminal, use --json + jq."""
    if cmd == "sync":
        print(f"Synced {r['source']}")
        print(f"  ref: {r['ref_requested']} -> {r['ref_resolved']}")
        sig = r.get("signature") or {}
        if sig:
            glyph = "[ok]" if sig.get("status") == "valid" else (
                "[invalid]" if sig.get("status") == "invalid" else "[..]")
            enforced = " (enforced)" if r.get("signature_enforced") else ""
            by = f" -- {sig['signed_by']}" if sig.get("signed_by") else ""
            print(f"  signature: {glyph} {sig.get('status')}{enforced} ({sig.get('reason')}){by}")
        print(f"  total: {r['total']} | synced: {r['synced']} | invalid: {r['invalid']} | errored: {r['errored']}")
        for s in r["skills"]:
            icon = "[ok]" if s["status"] == "synced" else ("[invalid]" if s["status"] == "invalid" else "[err]")
            print(f"  {icon} {s['id']}")
    elif cmd == "query":
        print(f"Top {len(r['hits'])} skills for: \"{r['intent']}\"")
        print(f"  model: {r['embedding_model']}\n")
        for i, h in enumerate(r["hits"], start=1):
            print(f"  {i}. [{h['score']:.3f}] {h['identity']}")
            print(f"      {h['title']}")
    elif cmd == "bench":
        n = r["total"]
        pct = lambda x: f"{x/n*100:.1f}%" if n else "n/a"
        print(f"Bench against {n} queries")
        print(f"  truth: {r['truth_file']}")
        print(f"  model: {r['embedding_model']}")
        print(f"  top-1: {r['top1']}/{n} ({pct(r['top1'])})")
        print(f"  top-3: {r['top3']}/{n} ({pct(r['top3'])})")
        print(f"  top-{r['k']}: {r['topK']}/{n} ({pct(r['topK'])})")
        print(f"  mean top-1 score: {r['mean_top1_score']:.3f}")
        print(f"  mean margin (top-1 -> top-2): +{r['mean_margin']:.3f}")
        if r["failures"]:
            print(f"\nFailures ({len(r['failures'])}):")
            for f in r["failures"]:
                rank = f["rank"] if f["rank"] is not None else f">{r['k']}"
                exp_score = f["expected_score"] if f["expected_score"] is not None else "n/a"
                print(f"  FAIL \"{f['intent']}\"")
                print(f"      expected: {f['expected']} (rank {rank}, score {exp_score})")
                print(f"      got:      {f['got_top1']} (score {f['top1_score']:.3f})")


if __name__ == "__main__":
    sys.exit(main())
