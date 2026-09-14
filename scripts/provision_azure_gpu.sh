#!/usr/bin/env bash
# Provision an Azure A10 validation box, driver included.
#
# Encodes constraints that each cost real time to discover. See
# docs/gpu-validation-a10.md for the evidence behind every one:
#
#   * NVadsA10_v5 needs a GRID/vGPU driver, and Azure supports exactly one version for it
#     (vGPU 18.8 = 570.237). v17.x is EOL and lands on a v20.x host as a Code 43 failure.
#   * That driver is the CUDA 12.8 branch, which caps the software stack at
#     torch 2.10.0+cu128 and vLLM 0.19.1 (later vLLM wheels link libcudart.so.13 and need a
#     >= 580 driver that does not exist for this SKU).
#   * Security type MUST be Standard. Trusted Launch enables Secure Boot, which blocks the
#     unsigned GRID kernel module.
#   * Deploy NON-ZONALLY. The SKU is marked NotAvailableForSubscription in southeastasia, but
#     the restriction is type=Zone, so a regional deployment succeeds where a zonal one fails.
#   * Only the full-GPU sizes are usable. NV6/12/18ads are partitioned A10s (4/8/12 GiB).
#   * The default 30 GB OS disk cannot hold one checkpoint plus a CUDA wheel stack.
#
# Usage:
#   ./provision_azure_gpu.sh                       # 1x A10  (Standard_NV36ads_A10_v5)
#   ./provision_azure_gpu.sh --gpus 2              # 2x A10  (Standard_NV72ads_A10_v5)
#   ./provision_azure_gpu.sh --gpus 2 --spot       # ~5x cheaper, evictable
set -euo pipefail

RG="${RG:-rg-infertune}"
LOC="${LOC:-southeastasia}"
VM="${VM:-infertune}"
DISK_GB="${DISK_GB:-128}"
GPUS=1
SPOT=()
KEY_FILE="${KEY_FILE:-/projects/sandbox/.ops/id_ed25519.pub}"
GRID_URL="https://download.microsoft.com/download/5e213ec5-834f-4b0a-87f7-772751353b06/NVIDIA-Linux-x86_64-570.237-grid-azure.run"

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --spot) SPOT=(--priority Spot --max-price -1 --eviction-policy Deallocate); shift ;;
    --rg)   RG="$2"; shift 2 ;;
    --loc)  LOC="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$GPUS" in
  1) SIZE=Standard_NV36ads_A10_v5; VCPU=36 ;;
  2) SIZE=Standard_NV72ads_A10_v5; VCPU=72 ;;
  *) echo "only 1 or 2 GPUs are available in this series" >&2; exit 2 ;;
esac

echo "==> size $SIZE  (${GPUS}x A10, ${VCPU} vCPU) in $LOC"

# Fail early and legibly on the two things that actually block deployment.
QUOTA=$(az vm list-usage -l "$LOC" -o json \
  | python3 -c "import json,sys;print(next((u['limit'] for u in json.load(sys.stdin) if 'NVADSA10' in u['localName'].upper()),0))")
if [ "$QUOTA" -lt "$VCPU" ]; then
  echo "ERROR: NVADSA10v5 quota in $LOC is $QUOTA vCPUs, need $VCPU." >&2
  echo "Request an increase under Quotas -> Compute for 'Standard NVADSA10v5 Family'." >&2
  exit 1
fi
echo "    quota ok: $QUOTA vCPUs"

MYIP=$(curl -fsS --max-time 10 https://api.ipify.org)
echo "    locking SSH to $MYIP"

az group create -n "$RG" -l "$LOC" \
  --tags purpose=infertune-gpu-validation ephemeral=true -o none
az network vnet create -g "$RG" -n vnet-infertune -l "$LOC" \
  --address-prefixes 10.42.0.0/16 --subnet-name snet-gpu --subnet-prefixes 10.42.1.0/24 -o none
az network nsg create -g "$RG" -n nsg-infertune -l "$LOC" -o none
az network nsg rule create -g "$RG" --nsg-name nsg-infertune -n allow-ssh-sandbox \
  --priority 1000 --access Allow --protocol Tcp --direction Inbound \
  --source-address-prefixes "$MYIP" --destination-port-ranges 22 -o none
az network nsg rule create -g "$RG" --nsg-name nsg-infertune -n deny-all-inbound \
  --priority 4096 --access Deny --protocol '*' --direction Inbound \
  --source-address-prefixes '*' --destination-port-ranges '*' -o none
az network vnet subnet update -g "$RG" --vnet-name vnet-infertune -n snet-gpu \
  --network-security-group nsg-infertune -o none
az network public-ip create -g "$RG" -n pip-infertune -l "$LOC" \
  --sku Standard --allocation-method Static -o none

# No --zone: non-zonal placement, i.e. "No infrastructure redundancy required".
az vm create -g "$RG" -n "$VM" -l "$LOC" \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size "$SIZE" \
  --admin-username azureuser --ssh-key-values "$(cat "$KEY_FILE")" \
  --vnet-name vnet-infertune --subnet snet-gpu \
  --public-ip-address pip-infertune --nsg "" \
  --os-disk-size-gb "$DISK_GB" --storage-sku Premium_LRS \
  --security-type Standard \
  --os-disk-delete-option Delete --nic-delete-option Delete \
  "${SPOT[@]}" -o none

az vm auto-shutdown -g "$RG" -n "$VM" --time 0200 -o none || \
  echo "    warning: auto-shutdown not set; remember to deallocate"

IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
echo "==> VM up at $IP"

echo "==> waiting for SSH"
for _ in $(seq 1 40); do
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes \
      -i "${KEY_FILE%.pub}" "azureuser@$IP" true 2>/dev/null && break
  sleep 5
done

echo "==> installing GRID 570.237 and the pinned software stack"
ssh -o StrictHostKeyChecking=accept-new -i "${KEY_FILE%.pub}" "azureuser@$IP" \
    "GRID_URL='$GRID_URL' bash -s" <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# Secure Boot must be off for the unsigned GRID module; --security-type Standard ensures it.
sudo apt-get install -y -qq mokutil >/dev/null 2>&1 || true
echo "secure boot: $(mokutil --sb-state 2>&1 || echo 'n/a (Standard security type)')"

# Purge any pre-existing driver, then install fresh.
sudo apt-get purge -y -qq '^nvidia-.*' '^libnvidia-.*' '^cuda-.*' >/dev/null 2>&1 || true
sudo apt-get autoremove -y -qq >/dev/null 2>&1 || true
[ -x /usr/bin/nvidia-uninstall ] && sudo /usr/bin/nvidia-uninstall --silent || true
printf 'blacklist nouveau\noptions nouveau modeset=0\n' \
  | sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
sudo update-initramfs -u >/dev/null 2>&1

sudo apt-get update -qq
sudo apt-get install -y -qq build-essential dkms pkg-config libglvnd-dev \
  "linux-headers-$(uname -r)" >/dev/null 2>&1

RUN=$(basename "$GRID_URL")
[ -f "$RUN" ] || wget -q -O "$RUN" "$GRID_URL"
chmod +x "$RUN"
sudo ./"$RUN" --silent --dkms --no-cc-version-check >/dev/null 2>&1
sudo modprobe nvidia || true
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

# The one software stack that works on a 570 (CUDA 12.8) driver.
curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv >/dev/null 2>&1
source .venv/bin/activate
uv pip install --quiet "vllm==0.19.1" --torch-backend=cu128
python -c "
import torch, vllm
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| available', torch.cuda.is_available())
print('vllm ', vllm.__version__, '| devices', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'  gpu{i}: {p.name} sm_{p.major}{p.minor} '
          f'total {p.total_memory/1024**3:.2f} GiB, free {free/1024**3:.2f} GiB, {p.multi_processor_count} SMs')
"
REMOTE

echo
echo "==> ready.  ssh -i ${KEY_FILE%.pub} azureuser@$IP"
echo "    teardown:  az group delete -n $RG --yes --no-wait"
