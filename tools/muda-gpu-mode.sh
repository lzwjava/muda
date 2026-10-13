#!/usr/bin/env bash
#
# muda-gpu-mode.sh - switch the Tesla M40 between the proprietary nvidia
# driver and the open nouveau driver.
#
#   sudo tools/muda-gpu-mode.sh status    # show current state (no root needed)
#   sudo tools/muda-gpu-mode.sh nouveau   # unbind nvidia, load nouveau
#   sudo tools/muda-gpu-mode.sh nvidia    # unbind nouveau, load nvidia
#
# SAFETY
#   * Unbinding is runtime-only: a reboot always restores the nvidia driver
#     (nouveau is blacklisted in /etc/modprobe.d/blacklist-nouveau.conf).
#   * Anything using the M40 (ComfyUI, llama.cpp, X/PRIME) must be stopped
#     before switching, otherwise the unbind/rmmod will fail.
#   * The M40 has no display outputs; the desktop runs on the Intel i915 GPU.
#
set -uo pipefail

BDF="${MUDA_BDF:-0000:01:00.0}"
CMD="${1:-status}"

bind_of() {
    local link="/sys/bus/pci/devices/$BDF/driver"
    if [ -e "$link" ]; then
        basename "$(readlink "$link")"
    else
        echo "none"
    fi
}

module_loaded() { lsmod | awk -v m="$1" '$1 == m { print "yes"; exit }'; }

status() {
    echo "device            : $BDF"
    echo "bound driver      : $(bind_of)"
    echo "nvidia loaded     : $(module_loaded nvidia)"
    echo "nouveau loaded    : $(module_loaded nouveau)"
    echo "render nodes      :"
    for n in /dev/dri/renderD*; do
        [ -e "$n" ] || continue
        echo "    $n"
    done
    echo "PCI resource sizes:"
    local n=0 start end flags
    while read -r start end flags; do
        if [ -n "${end:-}" ] && [ "$end" != "$start" ]; then
            local s=$((16#${start#0x})) e=$((16#${end#0x}))
            echo "    BAR$n: $(( (e - s + 1) / 1048576 )) MiB"
        fi
        n=$((n + 1))
        [ "$n" -ge 3 ] && break
    done < "/sys/bus/pci/devices/$BDF/resource" 2>/dev/null
}

unbind() {
    local link="/sys/bus/pci/devices/$BDF/driver"
    if [ -e "$link" ]; then
        echo "unbinding $(basename "$(readlink "$link")") from $BDF"
        echo "$BDF" > "$link/unbind" || echo "  (unbind failed)"
    fi
}

teardown_nvidia() {
    # Ask the driver to release the GPU, then remove modules bottom-up.
    for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
        if [ -n "$(module_loaded "$m")" ]; then
            echo "rmmod $m"
            rmmod "$m" 2>/dev/null || echo "  ($m is in use; stop GPU apps first)"
        fi
    done
}

to_nouveau() {
    echo "=== switching $BDF to nouveau ==="
    unbind
    teardown_nvidia
    echo "modprobe nouveau"
    modprobe nouveau || { echo "modprobe nouveau failed"; exit 1; }
    sleep 2
    echo
    status
    echo
    echo "Try: MUDA_DRIVER=nouveau build/mudainfo"
}

to_nvidia() {
    echo "=== switching $BDF back to nvidia ==="
    unbind
    if [ -n "$(module_loaded nouveau)" ]; then
        echo "rmmod nouveau"
        rmmod nouveau 2>/dev/null || true
    fi
    for m in nvidia nvidia_modeset nvidia_drm nvidia_uvm; do
        echo "modprobe $m"
        modprobe "$m" 2>/dev/null || true
    done
    sleep 2
    echo
    status
}

case "$CMD" in
    status)  status ;;
    nouveau) to_nouveau ;;
    nvidia)  to_nvidia ;;
    *) echo "usage: $0 {status|nouveau|nvidia}"; exit 2 ;;
esac
