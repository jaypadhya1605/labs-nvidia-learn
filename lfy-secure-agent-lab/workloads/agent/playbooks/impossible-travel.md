# Playbook: impossible travel
Check SigninLogs for the user across 24 hours. Compare IPs against ThreatIntel. A malicious-reputation IP or a non-corporate
ASN after a successful sign-in is a true positive: recommend session revocation and a password reset (human approval required).
Corporate VPN egress (ThreatIntel source kha-it) explains most benign cases.
