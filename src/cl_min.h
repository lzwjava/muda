/*
 * cl_min.h - minimal OpenCL 1.2 declarations.
 *
 * muda does not depend on the OpenCL headers or the libOpenCL.so development
 * symlink.  We declare just enough of the ABI to load libOpenCL.so.1 with
 * dlopen() at run time.  This keeps the build dependency-free and lets muda
 * degrade gracefully on machines without an OpenCL runtime.
 */
#ifndef MUDA_CL_MIN_H
#define MUDA_CL_MIN_H

#include <stddef.h>

typedef int                cl_int;
typedef unsigned int       cl_uint;
typedef long               cl_long;
typedef unsigned long      cl_ulong;
typedef unsigned long      cl_bitfield;
typedef void              *cl_platform_id;
typedef void              *cl_device_id;
typedef void              *cl_context;
typedef void              *cl_command_queue;
typedef void              *cl_program;
typedef void              *cl_kernel;
typedef void              *cl_mem;
typedef void              *cl_event;

typedef void (*cl_context_callback)(const char *errinfo, const void *private_info,
                                    size_t cb, void *user_data);
typedef void (*cl_program_callback)(cl_program program, void *user_data);

/* Error codes */
#define CL_SUCCESS                                  0

/* Platform / device info */
#define CL_PLATFORM_NAME                            0x0902
#define CL_PLATFORM_VENDOR                          0x0903
#define CL_PLATFORM_VERSION                         0x0901
#define CL_DEVICE_NAME                              0x102B
#define CL_DEVICE_VENDOR                            0x102C
#define CL_DEVICE_TYPE                              0x1000
#define CL_DEVICE_VERSION                           0x102F
#define CL_DRIVER_VERSION                           0x102D
#define CL_DEVICE_EXTENSIONS                        0x1030
#define CL_DEVICE_MAX_COMPUTE_UNITS                 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE               0x1004
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS          0x1003
#define CL_DEVICE_MAX_WORK_ITEM_SIZES               0x1005
#define CL_DEVICE_GLOBAL_MEM_SIZE                   0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE                    0x1023
#define CL_DEVICE_MAX_CLOCK_FREQUENCY               0x100C
#define CL_DEVICE_ADDRESS_BITS                      0x100D
#define CL_DEVICE_AVAILABLE                         0x1027
#define CL_DEVICE_COMPILER_AVAILABLE                0x1028

/* NVIDIA device attribute query extension */
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV            0x4002
#define CL_DEVICE_WARP_SIZE_NV                      0x4003
#define CL_DEVICE_GPU_OVERLAP_NV                    0x4004
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV            0x4005
#define CL_DEVICE_INTEGRATED_MEMORY_NV              0x4006
#define CL_DEVICE_PCI_BUS_ID_NV                     0x4008
#define CL_DEVICE_PCI_SLOT_ID_NV                    0x4009
#define CL_DEVICE_PCI_DOMAIN_ID_NV                  0x400A

/* Device types */
#define CL_DEVICE_TYPE_CPU                          (1 << 1)
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
#define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF

/* Memory flags */
#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_MEM_WRITE_ONLY                           (1 << 1)
#define CL_MEM_READ_ONLY                            (1 << 2)
#define CL_MEM_ALLOC_HOST_PTR                       (1 << 4)
#define CL_MEM_COPY_HOST_PTR                        (1 << 5)

/* Command queue flags */
#define CL_QUEUE_PROFILING_ENABLE                   (1 << 1)

/* Program build info */
#define CL_PROGRAM_BUILD_LOG                        0x1183

/* Profiling info */
#define CL_PROFILING_COMMAND_START                  0x1282
#define CL_PROFILING_COMMAND_END                    0x1283

/* Event info */
#define CL_EVENT_COMMAND_EXECUTION_STATUS           0x11D3
#define CL_QUEUED                                   0x2
#define CL_SUBMITTED                                0x3
#define CL_RUNNING                                  0x1
#define CL_COMPLETE                                 0x0

/* Kernel work-group info */
#define CL_KERNEL_WORK_GROUP_SIZE                   0x11B0
#define CL_KERNEL_COMPILE_WORK_GROUP_SIZE           0x11B1
#define CL_KERNEL_LOCAL_MEM_SIZE                    0x11B2
#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3
#define CL_KERNEL_PRIVATE_MEM_SIZE                  0x11B4

#define CL_TRUE  1
#define CL_FALSE 0

/* ------------------------------------------------------------------ */
/* Function pointer table                                              */
/* ------------------------------------------------------------------ */
typedef struct cl_api {
    cl_int (*GetPlatformIDs)(cl_uint num_entries, cl_platform_id *platforms,
                             cl_uint *num_platforms);
    cl_int (*GetPlatformInfo)(cl_platform_id platform, cl_uint param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret);
    cl_int (*GetDeviceIDs)(cl_platform_id platform, cl_bitfield device_type,
                           cl_uint num_entries, cl_device_id *devices,
                           cl_uint *num_devices);
    cl_int (*GetDeviceInfo)(cl_device_id device, cl_uint param_name,
                            size_t param_value_size, void *param_value,
                            size_t *param_value_size_ret);
    cl_context (*CreateContext)(const cl_int *properties, cl_uint num_devices,
                                const cl_device_id *devices,
                                cl_context_callback pfn_notify, void *user_data,
                                cl_int *errcode_ret);
    cl_command_queue (*CreateCommandQueue)(cl_context context, cl_device_id device,
                                           cl_bitfield properties, cl_int *errcode_ret);
    cl_program (*CreateProgramWithSource)(cl_context context, cl_uint count,
                                          const char **strings, const size_t *lengths,
                                          cl_int *errcode_ret);
    cl_int (*BuildProgram)(cl_program program, cl_uint num_devices,
                           const cl_device_id *device_list, const char *options,
                           cl_program_callback pfn_notify, void *user_data);
    cl_int (*GetProgramBuildInfo)(cl_program program, cl_device_id device,
                                  cl_uint param_name, size_t param_value_size,
                                  void *param_value, size_t *param_value_size_ret);
    cl_kernel (*CreateKernel)(cl_program program, const char *kernel_name,
                              cl_int *errcode_ret);
    cl_int (*ReleaseKernel)(cl_kernel kernel);
    cl_int (*ReleaseProgram)(cl_program program);
    cl_mem (*CreateBuffer)(cl_context context, cl_bitfield flags, size_t size,
                           void *host_ptr, cl_int *errcode_ret);
    cl_int (*ReleaseMemObject)(cl_mem buffer);
    cl_int (*SetKernelArg)(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                           const void *arg_value);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue queue, cl_kernel kernel,
                                   cl_uint work_dim, const size_t *global_work_offset,
                                   const size_t *global_work_size,
                                   const size_t *local_work_size, cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list, cl_event *event);
    cl_int (*EnqueueReadBuffer)(cl_command_queue queue, cl_mem buffer, cl_uint blocking,
                                size_t offset, size_t size, void *ptr,
                                cl_uint num_events, const cl_event *wait_list, cl_event *event);
    cl_int (*EnqueueWriteBuffer)(cl_command_queue queue, cl_mem buffer, cl_uint blocking,
                                 size_t offset, size_t size, const void *ptr,
                                 cl_uint num_events, const cl_event *wait_list, cl_event *event);
    cl_int (*EnqueueCopyBuffer)(cl_command_queue queue, cl_mem src, cl_mem dst,
                                size_t src_offset, size_t dst_offset, size_t size,
                                cl_uint num_events, const cl_event *wait_list, cl_event *event);
    cl_int (*EnqueueFillBuffer)(cl_command_queue queue, cl_mem buffer, const void *pattern,
                                size_t pattern_size, size_t offset, size_t size,
                                cl_uint num_events, const cl_event *wait_list, cl_event *event);
    cl_int (*EnqueueMarkerWithWaitList)(cl_command_queue queue, cl_uint num_events,
                                        const cl_event *wait_list, cl_event *event);
    cl_int (*Finish)(cl_command_queue queue);
    cl_int (*Flush)(cl_command_queue queue);
    cl_int (*ReleaseCommandQueue)(cl_command_queue queue);
    cl_int (*ReleaseContext)(cl_context context);
    cl_int (*WaitForEvents)(cl_uint num_events, const cl_event *event_list);
    cl_int (*ReleaseEvent)(cl_event event);
    cl_int (*GetEventProfilingInfo)(cl_event event, cl_uint param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret);
    cl_int (*GetEventInfo)(cl_event event, cl_uint param_name,
                           size_t param_value_size, void *param_value,
                           size_t *param_value_size_ret);
    cl_int (*GetKernelWorkGroupInfo)(cl_kernel kernel, cl_device_id device,
                                     cl_uint param_name, size_t param_value_size,
                                     void *param_value, size_t *param_value_size_ret);
} cl_api;

/* Loads libOpenCL.so.1 and fills the table.  Returns 0 on success. */
int  muda_cl_load(void);
void muda_cl_unload(void);
extern cl_api g_cl;
extern int    g_cl_loaded;

#endif /* MUDA_CL_MIN_H */
