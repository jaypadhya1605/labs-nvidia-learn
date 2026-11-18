# Sandbox policy intent for the SOC "code-writing analyst" (M10)

Write the intent first, then generate the concrete OpenShell policy and let the prover check it.

| Surface | Allow | Deny |
|---|---|---|
| Filesystem | read/write `/sandbox`; read-only system libraries | host home, `~/.ssh`, `~/.azure`, `~/.kube`, anything outside declared paths |
| Network | the inference route only (OpenShell provider for the guarded gateway) | everything else, including package indexes after build time |
| Credentials | none inside the sandbox; the provider injects them only for the approved endpoint | API keys in env vars or files |
| Process | unprivileged user, no new privileges | setuid, raw sockets, mounting |

How to turn this into YAML: install the OpenShell agent skills (`npx skills add NVIDIA/OpenShell`) and ask your coding
agent to use `generate-sandbox-policy` with this table, or follow the docs tutorial "first network policy". Then run the
policy advisor/prover on the change and confirm it grants no new credentialed host. Commit the YAML next to this file.
