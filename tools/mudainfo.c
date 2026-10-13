/*
 * mudainfo - print information about the device muda will use.
 *
 * The first half comes from the active muda driver.  The second half is a
 * small, driver-independent PCI probe that reads sysfs directly, which is
 * useful for confirming which physical card is behind the runtime.
 */
#include "muda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

static unsigned long long read_hex_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    unsigned long long v = 0;
    if (fscanf(f, "%llx", &v) != 1)
        v = 0;
    fclose(f);
    return v;
}

static void print_bar0_peek(const char *bdf)
{
    char path[1024];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        printf("    BAR0 register peek : requires root (%s)\n", strerror(errno));
        return;
    }
    unsigned int boot0 = 0;
    ssize_t n = pread(fd, &boot0, sizeof(boot0), 0x0);
    close(fd);
    if (n == (ssize_t)sizeof(boot0))
        printf("    NV_PMC_BOOT_0 @0x0  : 0x%08x\n", boot0);
    else
        printf("    BAR0 register peek : failed\n");
}

static void pci_probe(void)
{
    printf("PCI devices (vendor 0x10de, NVIDIA):\n");
    DIR *d = opendir("/sys/bus/pci/devices");
    if (!d) {
        printf("  (cannot open /sys/bus/pci/devices)\n");
        return;
    }
    struct dirent *ent;
    int found = 0;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.')
            continue;
        char base[300], path[1024];
        snprintf(base, sizeof(base), "/sys/bus/pci/devices/%s", ent->d_name);

        snprintf(path, sizeof(path), "%s/vendor", base);
        unsigned long long vendor = read_hex_file(path);
        if (vendor != 0x10de)
            continue;
        found = 1;

        snprintf(path, sizeof(path), "%s/device", base);
        unsigned long long dev = read_hex_file(path);
        snprintf(path, sizeof(path), "%s/class", base);
        unsigned long long cls = read_hex_file(path);

        printf("  %s  vendor=0x%04llx device=0x%04llx class=0x%06llx\n",
               ent->d_name, vendor, dev, cls);

        snprintf(path, sizeof(path), "%s/resource", base);
        FILE *rf = fopen(path, "r");
        if (rf) {
            char line[256];
            int idx = 0;
            while (fgets(line, sizeof(line), rf) && idx < 3) {
                unsigned long long start = 0, end = 0, flags = 0;
                if (sscanf(line, "%llx %llx %llx", &start, &end, &flags) == 3) {
                    if (end > start)
                        printf("    BAR%d: %016llx - %016llx  (%llu MiB)  flags=0x%llx\n",
                               idx, start, end, (end - start + 1) >> 20, flags);
                }
                idx++;
            }
            fclose(rf);
        }
        print_bar0_peek(ent->d_name);
    }
    if (!found)
        printf("  (none)\n");
    closedir(d);
}

int main(void)
{
    printf("muda %d.%d.%d\n", MUDA_VERSION_MAJOR, MUDA_VERSION_MINOR, MUDA_VERSION_PATCH);

    mudaError_t e = mudaInit(0);
    if (e != mudaSuccess) {
        printf("mudaInit failed: %s\n", mudaGetErrorString(e));
        return 1;
    }
    printf("driver: %s\n\n", mudaGetDriverName());

    int count = 0;
    e = mudaGetDeviceCount(&count);
    if (e != mudaSuccess)
        printf("mudaGetDeviceCount: %s\n", mudaGetErrorString(e));
    printf("device count: %d\n\n", count);

    for (int i = 0; i < count; i++) {
        mudaDeviceProp p;
        if (mudaGetDeviceProperties(&p, i) != mudaSuccess)
            continue;
        printf("Device %d: %s\n", i, p.name);
        printf("  compute capability : %d.%d\n", p.computeCapabilityMajor, p.computeCapabilityMinor);
        printf("  driver version     : %s\n", p.driverVersion);
        printf("  device version     : %s\n", p.deviceVersion);
        printf("  global memory      : %.0f MiB\n", (double)p.totalGlobalMem / 1048576.0);
        printf("  shared per block   : %zu bytes\n", p.sharedMemPerBlock);
        printf("  multiprocessors    : %d\n", p.multiProcessorCount);
        printf("  threads per block  : %d\n", p.maxThreadsPerBlock);
        printf("  warp size          : %d\n", p.warpSize);
        printf("  clock rate         : %.0f MHz\n", (double)p.clockRateKHz / 1000.0);
        printf("  PCI bus/slot/domain: %02x:%02x.%x\n",
               p.pciBusID, p.pciDeviceID >> 3, p.pciDeviceID & 7);
        printf("\n");
    }

    pci_probe();

    mudaShutdown();
    return 0;
}
