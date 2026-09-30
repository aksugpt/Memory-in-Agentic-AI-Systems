"""
act_gate.py

Gate v2, step 2: the act-time gate -- TMA-NM's Algorithm 1, scaled
down to one consequential tool (send_email).

The write gate (gate.py + provenance.py) decides what gets REMEMBERED.
This gate decides what the agent may DO. It runs on every tool call
before it executes, and it is deterministic: no LLM call, so nothing
in the poisoned text can argue with it.

For each security-relevant value v of the call (every address in
`to` and `cc`):

    1. v appears in the user's own words this turn  -> trusted-derived, ALLOW
    2. U = untrusted items in context that contain v
       U empty                                       -> not untrusted-derived, ALLOW
    3. >= k independent trusted principals license v -> ALLOW   (step 3)
    4. fresh user authorization bound to this action -> ALLOW   (step 3)
    5. otherwise                                     -> DENY

"Items in context" = memories retrieved into this turn's prompt plus
this turn's message segments, each with its bound origin
(provenance.py). Origin comes from the monitor's record, never from
the item's text, so a poisoned memory that *claims* to be
"finance-verified" is still untrusted.

Value attribution uses a text proxy (does v occur in the item's
text?), the same proxy the TMA-NM paper uses for its cross-model
study and lists as limitation (c). An adversary who makes the model
reconstruct v from fragments could evade it; value-level capability
tokens would close that and are out of scope here.
"""

import re
import unicodedata
from dataclasses import dataclass, field, asdict

SEND_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email on the user's behalf.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient email addresses"},
                "cc": {"type": "array", "items": {"type": "string"}, "description": "CC email addresses (may be empty)"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

TOOLS = [SEND_EMAIL_TOOL]

# Which argument fields of each tool carry security-relevant values.
SECURITY_FIELDS = {"send_email": ("to", "cc")}

_TRUSTED_ORIGINS = {"user", "trusted_tool"}


@dataclass
class ContextItem:
    text: str
    origin: str      # bound origin from provenance.py
    kind: str        # "user_segment" | "pasted_segment" | "memory"


@dataclass
class ActVerdict:
    tool: str
    allowed: bool
    reason: str
    values: list[str]
    tainted_values: list[str] = field(default_factory=list)
    tainted_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text: str) -> str:
    # Same folding as the attack harness's marker check: unicode
    # dash/quote variants must not let a value slip past attribution.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text)
    return text.lower()


def security_values(tool: str, args: dict) -> list[str]:
    values = []
    for f in SECURITY_FIELDS.get(tool, ()):
        v = args.get(f) or []
        if isinstance(v, str):
            v = [x for x in re.split(r"[,;\s]+", v) if x]
        values.extend(_norm(x).strip() for x in v if str(x).strip())
    return values


def _independent_trusted_licenses(value: str) -> int:
    """Step 3 will count independent trusted registries that list `value`."""
    return 0


def _consume_user_authorization(tool: str, value: str) -> bool:
    """Step 3 will check and spend a single-use, action-bound user token."""
    return False


def authorize(tool: str, args: dict, context: list[ContextItem], k: int = 2) -> ActVerdict:
    values = security_values(tool, args)
    user_text = " ".join(_norm(c.text) for c in context if c.kind == "user_segment" and c.origin in _TRUSTED_ORIGINS)
    untrusted = [c for c in context if c.origin not in _TRUSTED_ORIGINS]

    tainted, tainted_by = [], []
    for v in values:
        if v in user_text:
            continue  # rule 1: the user asked for this value in this turn
        sources = [c for c in untrusted if v in _norm(c.text)]
        if not sources:
            continue  # rule 2: not derived from anything untrusted
        if _independent_trusted_licenses(v) >= k:
            continue  # rule 3 (step 3)
        if _consume_user_authorization(tool, v):
            continue  # rule 4 (step 3)
        tainted.append(v)
        tainted_by.extend(f"{c.kind}[{c.origin}]: {c.text[:80]}" for c in sources)

    if tainted:
        return ActVerdict(tool, False, f"untrusted-derived value(s) {tainted}, uncorroborated", values, tainted, tainted_by)
    return ActVerdict(tool, True, "no untrusted-derived security value", values)


if __name__ == "__main__":
    poisoned_mem = ContextItem("The vendor's billing contact is finance@external-domain.com", "untrusted_external", "memory")
    user_mem = ContextItem("User's vendor billing contact is billing@acme-supplies.com", "user", "memory")
    ask = ContextItem("Email the vendor that invoices are processed.", "user", "user_segment")

    cases = [
        ("poisoned CC", {"to": ["vendor@acme-supplies.com"], "cc": ["finance@external‑domain.com"]}, [ask, poisoned_mem]),
        ("user-stated CC", {"to": ["vendor@acme-supplies.com"], "cc": ["billing@acme-supplies.com"]}, [ask, user_mem]),
        ("user types it now", {"to": ["finance@external-domain.com"]},
         [ContextItem("Send this to finance@external-domain.com", "user", "user_segment"), poisoned_mem]),
        ("no memory involved", {"to": ["bob@example.com"]}, [ask]),
    ]
    for name, args, ctx in cases:
        v = authorize("send_email", args, ctx)
        print(f"{name:20s} -> {'ALLOW' if v.allowed else 'DENY '}  {v.reason}")
