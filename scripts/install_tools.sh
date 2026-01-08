#!/usr/bin/env bash
set -euo pipefail

# Installer for Hawx Recon Agent dependencies
# Run locally: chmod +x scripts/install_tools.sh && ./scripts/install_tools.sh
# This script will prompt for your sudo password.

APT_PKGS=(
  nmap
  curl
  whatweb
  dirb
  gobuster
  openvpn
  python3-pip
  ffuf
  nikto
  swaks
)

echo "Updating apt and installing packages: ${APT_PKGS[*]}"
sudo apt update
sudo apt install -y "${APT_PKGS[@]}"

# Ensure Go is installed
if command -v go >/dev/null 2>&1; then
  echo "Go already installed: $(go version)"
else
  echo "Installing golang"
  sudo apt install -y golang
fi

# Setup GOPATH and GOBIN
export GOPATH="$HOME/go"
export GOBIN="$GOPATH/bin"
mkdir -p "$GOBIN"

echo "Installing subfinder and httpx via 'go install'"
GOBIN="$GOBIN" go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
GOBIN="$GOBIN" go install github.com/projectdiscovery/httpx/cmd/httpx@latest

# Symlink go binaries to /usr/local/bin for convenience
sudo ln -sf "$GOBIN/subfinder" /usr/local/bin/subfinder || true
sudo ln -sf "$GOBIN/httpx" /usr/local/bin/httpx || true

# Clone exploitdb and symlink searchsploit
if [ -d "/opt/exploitdb" ]; then
  echo "/opt/exploitdb exists, pulling latest"
  sudo git -C /opt/exploitdb pull || true
else
  echo "Cloning exploitdb to /opt/exploitdb"
  sudo git clone https://github.com/offensive-security/exploitdb.git /opt/exploitdb
fi
sudo ln -sf /opt/exploitdb/searchsploit /usr/local/bin/searchsploit || true

# Final report
echo "\nInstalled tools (locations):"
for t in nmap curl whatweb subfinder dirb ffuf gobuster searchsploit openvpn nikto swaks httpx; do
  printf "%-12s %s\n" "$t" "$(command -v $t || echo 'not found')"
done

echo "\nDone. If any tool is 'not found', check your PATH or rerun the script." 
