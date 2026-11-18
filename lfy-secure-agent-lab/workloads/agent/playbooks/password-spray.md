# Playbook: password spray
Many accounts failing from one IP within minutes (ResultType 50126) is a password spray: true positive. Block the source IP at
the edge (human approval required). One account failing repeatedly from a known corporate IP is usually a user error.
