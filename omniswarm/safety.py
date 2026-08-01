"""Lightweight safety screen for the trust layer.

Two guards, both deliberately high-precision (few false positives) rather than
exhaustive — they make the *verdict* reflect safety, they are not a content
filter or a guarantee:

  * screen_request(text) — flags a clearly-harmful REQUEST by category so the
    engine can escalate it (and skip generation) instead of returning pass. This
    fixes the gap where harmful asks were judged only on quality and could pass.
  * reveals_internal_prompt(text) — catches an ANSWER that echoed OmniSwarm's own
    council/judge system prompts (prompt-exfiltration), so it can be withheld.

Disable with OMNISWARM_SAFETY_SCREEN=0 (e.g. for security research).
"""
import os
import re

_ENABLED = os.environ.get("OMNISWARM_SAFETY_SCREEN", "1") not in ("0", "false", "False")

# category -> pattern. Each requires a harmful INTENT verb near a harmful OBJECT,
# so educational/analytical questions ("explain how ransomware works", "how is
# rental income taxed") do not match.
_RULES: list[tuple[str, re.Pattern]] = [
    ("weapons/CBRN", re.compile(
        r"\b(synthesi[sz]e|make|produce|manufacture|build|construct|create)\b.{0,45}"
        r"\b(nerve agent|sarin|vx gas|tabun|mustard gas|bioweapon|biological weapon|"
        r"chemical weapon|dirty bomb|pipe bomb|explosive device|ied|nuclear (?:bomb|device|weapon)|"
        r"ricin|anthrax|chlorine gas)\b", re.I)),
    ("malware", re.compile(
        r"\b(write|create|build|generate|code|develop|make|design)\b.{0,45}"
        r"\b(ransomware|keylogger|spyware|trojan|rootkit|botnet|computer virus|"
        r"self-replicating worm|malware|credential stealer)\b", re.I)),
    ("phishing/credential-theft", re.compile(
        r"\b(phishing|spear.?phish\w*)\b.{0,45}\b(email|page|site|website|kit|campaign|link)\b"
        r"|\b(steal|harvest|capture|phish)\b.{0,35}\b(password|credential|login details|banking)\b"
        r"|\bimpersonat\w*\b.{0,45}\b(bank|paypal|irs|microsoft|apple|amazon|coinbase)\b", re.I)),
    ("fraud/deception", re.compile(
        r"\b(generate|write|create|produce|post|leave)\b.{0,35}\b(fake|fraudulent|bogus)\b.{0,45}"
        r"\b(reviews?|testimonials?|accounts?|identit\w+)\b"
        r"|\blaunder(?:ing)?\b.{0,20}\bmoney\b"
        r"|\b(hide|conceal|evade)\b.{0,40}\b(income|taxes?)\b.{0,25}\b(irs|hmrc|tax(?:es|man)?|"
        r"without getting caught)\b", re.I)),
]

# distinctive phrases lifted from council.py's internal system prompts. If an answer
# contains one, the model regurgitated our instructions rather than answering.
_INTERNAL_FINGERPRINTS = (
    "you are the council chair",
    "role-based critiques from several reviewers",
    "respond only with json",
    "you are a strict, referenceless judge",
    "you are a reviewer on a council",
)


def enabled() -> bool:
    return _ENABLED


def screen_request(text: str) -> str | None:
    """Return a harmful-category label if the request should be refused/escalated."""
    if not _ENABLED or not text:
        return None
    for category, pattern in _RULES:
        if pattern.search(text):
            return category
    return None


def reveals_internal_prompt(text: str) -> bool:
    """True if an answer leaked one of OmniSwarm's internal system prompts."""
    if not text:
        return False
    low = text.lower()
    return any(fp in low for fp in _INTERNAL_FINGERPRINTS)
