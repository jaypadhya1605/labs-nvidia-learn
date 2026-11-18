# Playbook: encoded PowerShell
PowerShell -enc launched by Office applications (winword.exe, excel.exe) is a classic initial-access pattern: true positive.
Check DeviceNetworkEvents on the host for connections right after launch. Signed scripts launched by ccmexec.exe (SCCM) are
usually benign. Recommend host isolation for true positives (human approval required).
