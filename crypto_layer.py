"""
crypto_layer.py

Gate v2, step 4: TMA-NM's M4 (tamper-evident verdict log) plus HMAC
signing of memory provenance.

1. AuditLog -- an append-only JSONL hash chain. Every write decision,
   act-gate verdict, hold and user confirmation is one entry:

       hash_i = SHA256(hash_{i-1} || canonical(entry_i))
       mac_i  = HMAC(key, hash_i)

   Editing, deleting or reordering any earlier entry breaks every
   hash after it. The HMAC means someone with write access to the file
   but not the key can't simply recompute the chain. As in the paper,
   this gives auditability, not enforcement: it changes no decisions.

   A hash chain alone can't detect the newest entries being truncated
   off the end. verify(expected_head=...) closes that when the latest
   head hash has been anchored somewhere else (printed, emailed,
   committed).

2. sign_memory / verify_memory -- an HMAC over the fields that carry
   authority (text, source, origin, act_class), stored in the memory's
   metadata. Origin binding (M1) is only as good as the label at rest:
   if someone edits the vector store to relabel a poisoned memory as
   origin "user", the signature no longer matches and Gate v2 treats
   the memory as untrusted (fail-safe).

Key management: set MEMORY_SIGNING_KEY in .env (any long random
string). Without it, a random key is generated once and saved next to
the data (<memory_path>/.signing_key) so separate sessions can verify
each other's writes. That fallback is fine for experiments, but a key
stored beside the data doesn't protect against someone who can read
that folder. Use the env var for anything real.
"""

import hashlib
import hmac
import json
import os
import secrets
import time

GENESIS = "0" * 64
SIGNED_MEMORY_FIELDS = ("source", "origin", "act_class")


def load_key(memory_path: str) -> bytes:
    env = os.environ.get("MEMORY_SIGNING_KEY")
    if env:
        return env.encode()
    os.makedirs(memory_path, exist_ok=True)
    key_file = os.path.join(memory_path, ".signing_key")
    if not os.path.exists(key_file):
        with open(key_file, "w", encoding="utf-8") as f:
            f.write(secrets.token_hex(32))
    with open(key_file, encoding="utf-8") as f:
        return f.read().strip().encode()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str).encode()


# ---------------------------------------------------------------- memory signing

def sign_memory(key: bytes, text: str, meta: dict) -> str:
    payload = {"text": text, **{f: meta.get(f) for f in SIGNED_MEMORY_FIELDS}}
    return hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()


def verify_memory(key: bytes, text: str, meta: dict) -> bool:
    sig = meta.get("sig")
    return bool(sig) and hmac.compare_digest(sig, sign_memory(key, text, meta))


# ---------------------------------------------------------------- audit log

class AuditLog:
    def __init__(self, path: str, key: bytes):
        self.path = path
        self.key = key
        self.seq, self.head = 0, GENESIS
        if os.path.exists(path):
            last = None
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last = line
            if last:
                entry = json.loads(last)
                self.seq, self.head = entry["seq"] + 1, entry["hash"]

    def append(self, event: str, data: dict) -> str:
        """O(1): one hash over the previous head, one HMAC. No model call."""
        body = {"seq": self.seq, "ts": time.time(), "event": event, "data": data, "prev": self.head}
        digest = hashlib.sha256(self.head.encode() + _canonical(body)).hexdigest()
        mac = hmac.new(self.key, digest.encode(), hashlib.sha256).hexdigest()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**body, "hash": digest, "mac": mac}, default=str) + "\n")
        self.seq, self.head = self.seq + 1, digest
        return digest


def verify(path: str, key: bytes, expected_head: str | None = None) -> tuple[bool, str]:
    """
    Walk the chain from genesis. Returns (ok, message); on failure the
    message names the first bad entry.
    """
    prev, seq = GENESIS, 0
    if not os.path.exists(path):
        return False, f"no log at {path}"
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False, f"line {lineno}: not valid JSON"
            body = {k: entry.get(k) for k in ("seq", "ts", "event", "data", "prev")}
            if entry.get("seq") != seq:
                return False, f"line {lineno}: seq {entry.get('seq')} (expected {seq}) -- entry deleted or reordered"
            if entry.get("prev") != prev:
                return False, f"seq {seq}: prev-hash mismatch -- chain broken before this entry"
            digest = hashlib.sha256(prev.encode() + _canonical(body)).hexdigest()
            if digest != entry.get("hash"):
                return False, f"seq {seq}: content does not match its hash -- entry edited"
            mac = hmac.new(key, digest.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(mac, entry.get("mac", "")):
                return False, f"seq {seq}: bad HMAC -- chain rewritten without the key"
            prev, seq = digest, seq + 1
    if expected_head is not None and prev != expected_head:
        return False, f"head {prev[:12]}... != anchored head {expected_head[:12]}... -- entries truncated from the end"
    return True, f"ok: {seq} entries, head {prev[:12]}..."


if __name__ == "__main__":
    import shutil
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        # python crypto_layer.py verify <memory_path> [expected_head]
        mem_path = sys.argv[2]
        ok, msg = verify(os.path.join(mem_path, "audit_log.jsonl"), load_key(mem_path),
                         sys.argv[3] if len(sys.argv) > 3 else None)
        print(("VALID   " if ok else "TAMPERED ") + msg)
        raise SystemExit(0 if ok else 1)

    # Self-test: build a small chain, then tamper with it four ways.
    demo = "./chroma_crypto_selftest"
    shutil.rmtree(demo, ignore_errors=True)
    key = load_key(demo)
    log_path = os.path.join(demo, "audit_log.jsonl")
    log = AuditLog(log_path, key)
    for i in range(4):
        log.append("memory_write", {"text": f"memory {i}", "origin": "untrusted_external", "action": "quarantine"})
    anchored = log.head
    print("clean chain        ->", verify(log_path, key, anchored)[1])

    lines = open(log_path, encoding="utf-8").read().splitlines()

    def check(label, new_lines, k=key):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"{label:18s} ->", verify(log_path, k, anchored)[1])

    edited = json.loads(lines[1]); edited["data"]["origin"] = "user"
    check("edit an entry", [lines[0], json.dumps(edited)] + lines[2:])
    check("delete an entry", [lines[0]] + lines[2:])
    check("truncate the end", lines[:3])

    # Attacker without the key recomputes every hash after an edit:
    rewritten, prev = [], GENESIS
    for i, line in enumerate(lines):
        e = json.loads(line)
        if i == 1:
            e["data"]["origin"] = "user"
        e["prev"] = prev
        body = {k: e[k] for k in ("seq", "ts", "event", "data", "prev")}
        e["hash"] = hashlib.sha256(prev.encode() + _canonical(body)).hexdigest()
        prev = e["hash"]
        rewritten.append(json.dumps(e))
    check("recompute hashes", rewritten)

    meta = {"source": "processed_document", "origin": "untrusted_external", "act_class": "none"}
    meta["sig"] = sign_memory(key, "finance@external-domain.com", meta)
    print("\nsigned memory      ->", verify_memory(key, "finance@external-domain.com", meta))
    print("relabelled 'user'  ->", verify_memory(key, "finance@external-domain.com", {**meta, "origin": "user", "act_class": "act"}))
    shutil.rmtree(demo, ignore_errors=True)
