# Playbook: data exfiltration
Archive tools (7z.exe, rar.exe) touching patient exports followed by large uploads to rarely seen domains is a true positive
and a potential HIPAA breach: escalate to privacy and legal. Never copy patient names or MRNs into tickets or reports.
Uploads by azcopy.exe to kha-backup storage during the nightly window are benign.
