# Start here (Windows laptop + WSL2 Ubuntu)

All lab commands are bash and run inside **WSL2 Ubuntu**, from `~/nvidia-labs` on the Linux filesystem (not `/mnt/c`,
which is slow and breaks line endings).

1. Install WSL2 (PowerShell **as Administrator**, then reboot):
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Open **Ubuntu** from the Start menu and create your Linux user.
2. Copy the labs into your Linux home (Ubuntu terminal):
   ```bash
   cp -r /mnt/c/Users/jaypadhya/nvidia-labs ~/nvidia-labs
   ```
3. Install basics (Ubuntu terminal). Helm and kind follow the pins in `env/pins.env`:
   ```bash
   sudo apt update && sudo apt install -y python3 python3-venv python3-pip git gh jq unzip curl
   sudo snap install yq && sudo snap install kubectl --classic && sudo snap install terraform --classic
   sudo snap install google-cloud-cli --classic
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | DESIRED_VERSION=v3.22.1 bash
   curl -Lo kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-arm64 && sudo install kind /usr/local/bin/ && rm kind
   ```
4. Docker (needed from Day 5, M9 on kind). Docker Engine inside WSL, no Docker Desktop license needed:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER
   ```
   Close and reopen the Ubuntu terminal, then check with `docker run --rm hello-world`.
5. Open in VS Code (installs the WSL extension on first run): `code ~/nvidia-labs`
6. In Copilot Chat, pick **NVIDIA Lab Coach** in the Agent dropdown and type `start`.

Folders:
- `plv-multicloud-nim-lab/` - Lab 04, the centerpiece for JR2027588
- `ttg-agentic-genai-lab/` - Lab 05
- `kha-genai-inference-lab/` - Lab 02
- `lfy-secure-agent-lab/` - Lab 03
- `docs/` - all lab READMEs in one place + the NVIDIA SA strategy doc
- `.github/agents/nvidia-lab-coach.agent.md` - the coach (its 14-day plan matches your sprint doc)

Rules: personal cloud accounts only (never the company Azure subscription or work logins); WSL on this laptop;
synthetic data only; pause clusters every night.
