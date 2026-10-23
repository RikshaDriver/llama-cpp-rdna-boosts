#!/bin/bash
# §1 matrix driver for the r116 field/MoE arm (TODO #51).
#   ./matrix116.sh <part|full> <tensor|layer> <gpus> <gpus> ...
# Each cell = {none, draft-mtp(3), draft-mtp-adaptive(7)} x that (residency class, split, gpus set).
# Residency is held constant *per device* by sizing MOE_EXPERT_CACHE_MIB from the host-expert bytes,
# because the budget is MiB/device and under -sm tensor a device holds host_bytes/n_devices.
#   part: -ncmoe 48, host set 40925 MiB, target 92 %   (2 GPU 18830/device, 3 GPU 12550/device)
#   full: -ncmoe 48, auto arena = 100 % identity at 2 and 3 GPUs (-c 40000)
set -u
HERE=$(dirname "$0")
BUILD=${BUILD:-/home/stew675/llama.cpp/build-rocm}
CLASS=$1; SM=$2; shift 2
OUT=${OUT:-/tmp/srr/matrix116.csv}
case "$CLASS" in
  part) NCMOE=48; C2=18830; C3=12550;;
  full) NCMOE=48; C2=; C3=;;   # auto (the shipped default) = 100 % identity at both counts at -c 40000
  *) echo "usage: matrix116.sh <part|full> <tensor|layer> <gpus>..."; exit 2;;
esac
for g in "$@"; do
  case "$g" in
    0,1)   CACHE=$C2; ND=2;;
    0,1,2) CACHE=$C3; ND=3;;
    *) echo "bad gpu set $g"; exit 2;;
  esac
  tag=$(echo "$g" | tr -d ',')
  for spec in none draft-mtp draft-mtp-adaptive; do
    nmax=7; [ "$spec" = draft-mtp ] && nmax=3
    T="${CLASS}-${SM}-${ND}g-${spec}"
    GPUS=$g SM=$SM SPEC=$spec NMAX=$nmax NCMOE=$NCMOE CACHE=$CACHE N=1000 CTX=${CTX:-40000} TAG=$T \
      "$HERE/field116.sh" "$BUILD" | tee -a "$OUT"
  done
done
echo "=== matrix116 $CLASS $SM done $(date)"
