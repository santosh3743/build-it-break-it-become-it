"""
The two OWASP lists this threat model must cover, plus the MITRE ATLAS tactics
it tags threats with.

Keeping the catalogs in one file means a list revision is a one-file change,
and `validate.py` will immediately tell you which new item nothing treats yet.
"""

# OWASP Top 10 for Large Language Model Applications, 2025 edition
# (OWASP GenAI Security Project, released November 2024).
OWASP_LLM_2025 = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Supply Chain",
    "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM08": "Vector and Embedding Weaknesses",
    "LLM09": "Misinformation",
    "LLM10": "Unbounded Consumption",
}

# OWASP Top 10 for Agentic Applications, 2026 edition
# (OWASP GenAI Security Project, Agentic Security Initiative, December 2025).
# The "ASI" ids are the ones the published list uses. If a later revision
# renumbers or renames an item, change it here: every mapping below is by id,
# so the validator will flag anything that no longer lines up.
OWASP_AGENTIC_2026 = {
    "ASI01": "Agent Goal Hijack",
    "ASI02": "Tool Misuse and Exploitation",
    "ASI03": "Identity and Privilege Abuse",
    "ASI04": "Agentic Supply Chain Vulnerabilities",
    "ASI05": "Unexpected Code Execution",
    "ASI06": "Memory and Context Poisoning",
    "ASI07": "Insecure Inter-Agent Communication",
    "ASI08": "Cascading Failures",
    "ASI09": "Human-Agent Trust Exploitation",
    "ASI10": "Rogue Agents",
}

ALL_OWASP = {**OWASP_LLM_2025, **OWASP_AGENTIC_2026}

# MITRE ATLAS tactics used to tag threats. Only tactic NAMES are used here
# (ATLAS adapts the ATT&CK tactic set to AI systems); technique ids are cited
# in the README only for the two we are sure of.
ATLAS_TACTICS = (
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Collection",
    "Exfiltration",
    "Impact",
)


def owasp_label(item_id: str) -> str:
    return f"{item_id} {ALL_OWASP.get(item_id, '<unknown item>')}"
