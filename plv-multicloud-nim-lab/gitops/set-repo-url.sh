#!/usr/bin/env bash
# Point every Argo CD Application at YOUR fork (Argo CD cannot template the repo URL). Run once, then commit + push.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
[ "${GITHUB_OWNER}" != "your-github-user" ] || { echo "set GITHUB_OWNER in env/lab.env first"; exit 1; }
grep -rl "https://github.com/YOUR_GITHUB_USER/plv-multicloud-nim-lab.git" gitops | while read -r f; do
  sed "s#https://github.com/YOUR_GITHUB_USER/plv-multicloud-nim-lab.git#${GIT_REPO_URL}#g" "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  echo "updated $f"
done
echo "Now: git add gitops && git commit -m 'point GitOps at my fork' && git push"
