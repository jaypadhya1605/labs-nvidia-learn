# Windows setup: WSL2 on a Windows ARM64 laptop

Your laptop is **Windows on ARM64 with no NVIDIA GPU**, so the work splits two ways:
- **WSL2 Ubuntu** runs all the lab tooling: Terraform, kubectl, helm, Python tools and the quiz.
- **Cloud GPU nodes** run everything that needs a GPU: AKS (A100) and GKE (RTX PRO 6000).

Nothing in this capstone needs a local GPU.

> Use **personal** cloud accounts, keys and GitHub only: never a company subscription or work login. If this is
> your employer's laptop, check its policy before installing tools or running personal workloads on it.

## 1. WSL2 + Ubuntu 24.04 (once)
PowerShell **as Administrator**, then reboot:
```powershell
wsl --install -d Ubuntu-24.04
```
Open **Ubuntu** from the Start menu and create your Linux user. Ubuntu 24.04 ships Python 3.12, and Lab 04 needs 3.11
or newer. systemd is on by default, which `snap` needs.

## 2. Put the labs on the Linux filesystem
```bash
cp -r /mnt/c/Users/jaypadhya/nvidia-labs ~/nvidia-labs
cd ~/nvidia-labs && find . -name "*.sh" -exec sed -i 's/\r$//' {} +   # strip Windows line endings, if any
```
Work in `~/nvidia-labs`, not `/mnt/c/...`: `/mnt/c` is slow, and Windows editors can add CRLF line endings that
break bash.

Already copied the labs into WSL earlier? Then bring over only the capstone and its coach:
```bash
cp -r /mnt/c/Users/jaypadhya/nvidia-labs/jr2027588-capstone ~/nvidia-labs/
```

## 3. Tools (Ubuntu terminal; every one has an arm64 build)
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git gh jq unzip curl gettext-base   # gettext-base = envsubst
sudo snap install yq && sudo snap install kubectl --classic && sudo snap install terraform --classic
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | DESIRED_VERSION=v3.22.1 bash   # = PIN_HELM
```
Google Cloud CLI **with the GKE auth plugin**. Use the apt repo: the plugin is a separate package there.
```bash
sudo apt-get install -y apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin
```
ORAS (needed only in P8). The version matches `PIN_ORAS` in `plv-multicloud-nim-lab/env/pins.env`:
```bash
V=1.3.4; curl -LO https://github.com/oras-project/oras/releases/download/v${V}/oras_${V}_linux_arm64.tar.gz
mkdir -p oras-install && tar -zxf oras_${V}_linux_arm64.tar.gz -C oras-install && sudo mv oras-install/oras /usr/local/bin/
rm -rf oras_${V}_linux_arm64.tar.gz oras-install
```
Sign in:
```bash
az login && gcloud auth login && gcloud auth application-default login && gh auth login
```
Then `scripts/preflight.sh` in `plv-multicloud-nim-lab` lists anything still missing.

## 4. Docker (optional on this laptop)
The fast path never builds images locally (see section 5). If you want Docker anyway:
```bash
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER   # reopen the terminal afterwards
docker run --rm hello-world
```

## 5. The one ARM64 gotcha: the Plover API image must be linux/amd64
The cloud GPU nodes are x86_64. An image built natively on this laptop is arm64 and fails on the nodes with
`exec format error`. Two ways around it:
- **Recommended: let GitHub build it.** In your fork, `git tag v1.0.0 && git push origin v1.0.0`. The
  `.github/workflows/release.yml` workflow builds `ghcr.io/<you>/plover-api:1.0.0` for `linux/amd64` on GitHub's
  runner. Then make the GHCR package **public** (it contains no secrets, and the data is synthetic).
- **Local cross-build** (slower):
  ```bash
  docker run --privileged --rm tonistiigi/binfmt --install amd64
  docker buildx build --platform linux/amd64 -f app/Dockerfile -t ghcr.io/${GITHUB_OWNER_LC}/plover-api:1.0.0 --push .
  ```

## 6. Running the coach
**GitHub Copilot Chat** (VS Code + WSL extension):
1. Install the coach where Copilot looks for agents, then open the labs:
   ```bash
   mkdir -p ~/nvidia-labs/.github/agents
   cp ~/nvidia-labs/jr2027588-capstone/coach/jr2027588-coach.agent.md ~/nvidia-labs/.github/agents/
   code ~/nvidia-labs
   ```
2. In Copilot Chat, open the Agent dropdown and pick **JR2027588 Capstone Coach**.
3. Type `start`.

**Claude Code** (inside WSL):
```bash
cd ~/nvidia-labs/jr2027588-capstone && claude --add-dir ..
```
Then type `start`. `CLAUDE.md` makes it the same coach, and `--add-dir ..` lets it read the other labs.

**Knowledge sessions without any AI:**
- WSL: `python3 quiz/quiz.py`
- Windows PowerShell: `py quiz\quiz.py`

## 7. Keys
Each lab keeps secrets in its own `env/lab.env`, which is git-ignored. Don't keep keys in `.txt` files in the labs
folder. Delete `nvidia-labs/hugging-face-token.txt` and `nvidia-keys.txt` once the keys are in `env/lab.env`.
