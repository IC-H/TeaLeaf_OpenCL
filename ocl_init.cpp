#include "mpi.h"
#include "ocl_common.hpp"
#include "ocl_strings.hpp"

#include <sstream>
#include <iostream>
#include <algorithm>


void oclContextCallback(const char *errinfo, const void *, size_t, void *) {
    printf("Context callback: %s\n", errinfo);
}

// scoped_array: assumes pointer was allocated with operator new[]; destroys with operator delete[]
// Also supports allocation/reset with a number, which is the number of
// elements of type T.
template<typename T>
class scoped_array {
public:
    typedef scoped_array<T> this_type;

    scoped_array() : m_ptr(NULL) {}
    scoped_array(T *ptr) : m_ptr(NULL) { reset(ptr); }
    explicit scoped_array(size_t n) : m_ptr(NULL) { reset(n); }
    ~scoped_array() { reset(); }

    T *get() const { return m_ptr; }
    operator T *() const { return m_ptr; }
    T *operator ->() const { return m_ptr; }
    T &operator *() const { return *m_ptr; }
    T &operator [](int index) const { return m_ptr[index]; }

    this_type &operator =(T *ptr) { reset(ptr); return *this; }

    void reset(T *ptr = NULL) { delete[] m_ptr; m_ptr = ptr; }
    void reset(size_t n) { reset(new T[n]); }
    T *release() { T *ptr = m_ptr; m_ptr = NULL; return ptr; }

private:
    T *m_ptr;

    // noncopyable
    scoped_array(const this_type &);
    this_type &operator =(const this_type &);
};

CloverChunk chunk;

extern "C" void initialise_ocl_
(int* in_x_min, int* in_x_max,
 int* in_y_min, int* in_y_max)
{
    chunk = CloverChunk(in_x_min, in_x_max,
                        in_y_min, in_y_max);
}

// default ctor
CloverChunk::CloverChunk
(void)
{
    ;
}

extern "C" void timer_c_(double*);

CloverChunk::CloverChunk
(int* in_x_min, int* in_x_max,
 int* in_y_min, int* in_y_max)
:x_min(*in_x_min),
 x_max(*in_x_max),
 y_min(*in_y_min),
 y_max(*in_y_max)
{
#ifdef OCL_VERBOSE
    DBGOUT = stdout;
#else
    if (NULL == (DBGOUT = fopen("/dev/null", "w")))
    {
        DIE("Unable to open /dev/null to discard output\n");
    }
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0;
    timer_c_(&t0);

    if (!rank)
    {
        fprintf(stdout, "Initialising OpenCL\n");
    }

    initOcl();
    initProgram();
    initSizes();
    initReduction();
    initBuffers();
    initArgs();

    MPI_Barrier(MPI_COMM_WORLD);

    if (!rank)
    {
        double t1;
        timer_c_(&t1);

        fprintf(stdout, "Finished initialisation in %f seconds\n", t1-t0);
    }
}

static void stripString
(std::string & input_string)
{
    // trim whitespace from a string
    input_string.erase(input_string.find_last_not_of(" \n\r\t")+1);
    input_string.erase(input_string.begin(), input_string.begin()+input_string.find_first_not_of(" \n\r\t"));
}

static void listPlatforms
(std::vector<cl::Platform>& platforms)
{
    for (size_t pp = 0; pp < platforms.size(); pp++)
    {
        std::string profile, version, name, vendor;
        platforms.at(pp).getInfo(CL_PLATFORM_PROFILE, &profile);
        platforms.at(pp).getInfo(CL_PLATFORM_VERSION, &version);
        platforms.at(pp).getInfo(CL_PLATFORM_NAME, &name);
        platforms.at(pp).getInfo(CL_PLATFORM_VENDOR, &vendor);

        fprintf(stdout, "Platform %zu: %s - %s (profile = %s, version = %s)\n",
            pp, vendor.c_str(), name.c_str(), profile.c_str(), version.c_str());

        std::vector<cl::Device> devices;
        platforms.at(pp).getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (size_t ii = 0; ii < devices.size(); ii++)
        {
            std::string devname;
            cl_device_type dtype;
            devices.at(ii).getInfo(CL_DEVICE_NAME, &devname);
            devices.at(ii).getInfo(CL_DEVICE_TYPE, &dtype);
            stripString(devname);

            std::string dtype_str = strType(dtype);
            fprintf(stdout, " Device %zu: %s (%s)\n", ii, devname.c_str(), dtype_str.c_str());
        }
    }
}

void CloverChunk::initOcl
(void)
{
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    if(platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
        DIE("No platforms found\n");
    }

    // Read in from file - easier than passing in from fortran
    std::ifstream input("tea.in");
    input.exceptions(std::ifstream::badbit);

    if (!input.is_open())
    {
        // should never happen
        DIE("Input file not found\n");
    }

    // use first device whatever happens (ignore MPI rank) for running across different platforms
    bool usefirst = paramEnabled(input, "opencl_usefirst");

    profiler_on = paramEnabled(input, "profiler_on");

    std::string desired_vendor = readString(input, "opencl_vendor");

    int preferred_device = readInt(input, "opencl_device");
    preferred_device = (preferred_device < 0) ? 0 : preferred_device;
    fprintf(DBGOUT, "Preferred device is %d\n", preferred_device);

    std::string type_name = readString(input, "opencl_type");
    desired_type = typeMatch(type_name);

    int file_halo_depth = readInt(input, "halo_depth");

    halo_exchange_depth = file_halo_depth;

    if (halo_exchange_depth < 1)
    {
        DIE("Halo exchange depth unspecified or was too small");
    }

    bool tl_use_jacobi = paramEnabled(input, "tl_use_jacobi");
    bool tl_use_cg = paramEnabled(input, "tl_use_cg");
    bool tl_use_chebyshev = paramEnabled(input, "tl_use_chebyshev");
    bool tl_use_ppcg = paramEnabled(input, "tl_use_ppcg");

    // set solve
    if(!rank)fprintf(stdout, "Solver to use: ");
    if (tl_use_ppcg)
    {
        tea_solver = TEA_ENUM_PPCG;
        if(!rank)fprintf(stdout, "PPCG\n");
    }
    else if (tl_use_chebyshev)
    {
        tea_solver = TEA_ENUM_CHEBYSHEV;
        if(!rank)fprintf(stdout, "Chebyshev + CG\n");
    }
    else if (tl_use_cg)
    {
        tea_solver = TEA_ENUM_CG;
        if(!rank)fprintf(stdout, "Conjugate gradient\n");
    }
    else if (tl_use_jacobi)
    {
        tea_solver = TEA_ENUM_JACOBI;
        if(!rank)fprintf(stdout, "Jacobi\n");
    }
    else
    {
        tea_solver = TEA_ENUM_JACOBI;
        if(!rank)fprintf(stdout, "Jacobi (no solver specified in tea.in)\n");
    }

    std::string desired_preconditioner = readString(input, "tl_preconditioner_type");

    // set preconditioner type
    if(!rank)fprintf(stdout, "Preconditioner to use: ");
    if (desired_preconditioner.find("jac_diag") != std::string::npos)
    {
        preconditioner_type = TL_PREC_JAC_DIAG;
        if(!rank)fprintf(stdout, "Diagonal Jacobi\n");
    }
    else if (desired_preconditioner.find("jac_block") != std::string::npos)
    {
        preconditioner_type = TL_PREC_JAC_BLOCK;
        if(!rank)fprintf(stdout, "Block Jacobi\n");
    }
    else if (desired_preconditioner.find("none") != std::string::npos)
    {
        preconditioner_type = TL_PREC_NONE;
        if(!rank)fprintf(stdout, "None\n");
    }
    else
    {
        preconditioner_type = TL_PREC_NONE;
        if(!rank)fprintf(stdout, "None (no preconditioner specified in tea.in)\n");
    }

    // Query the available OpenCL devices.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

    // We'll just use the first device.
    device = devices[0];

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
    // checkError(status, "Failed to create context");

    // gets devices one at a time to prevent conflicts (on emerald)
    int ranks, cur_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    do
    {
        if (rank == cur_rank)
        {
            // choose reduction based on device type
            switch (desired_type)
            {
            case CL_DEVICE_TYPE_GPU : 
                device_type_prepro = "-DCL_DEVICE_TYPE_GPU ";
                break;
            case CL_DEVICE_TYPE_CPU : 
                device_type_prepro = "-DCL_DEVICE_TYPE_CPU ";
                break;
            case CL_DEVICE_TYPE_ACCELERATOR : 
                device_type_prepro = "-DCL_DEVICE_TYPE_ACCELERATOR ";
                break;
            default :
                device_type_prepro = "-DCL_DEVICE_TYPE_GPU ";
                break;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } while ((cur_rank++) < ranks);

    MPI_Barrier(MPI_COMM_WORLD);

    // Create the command queue.
    printf("Creating queue\n");
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    // checkError(status, "Failed to create command queue");
}

// Returns the platform name.
std::string CloverChunk::getPlatformName(cl_platform_id pid) {
    cl_int status;

    size_t sz;
    status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, 0, NULL, &sz);
    // checkError(status, "Query for platform name size failed");

    scoped_array<char> name(sz);
    status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, sz, name, NULL);
    // checkError(status, "Query for platform name failed");

    return name.get();
}

// Returns the list of all devices.
cl_device_id *CloverChunk::getDevices(cl_platform_id pid, cl_device_type dev_type, cl_uint *num_devices) {
    cl_int status;

    status = clGetDeviceIDs(pid, dev_type, 0, NULL, num_devices);
    // checkError(status, "Query for number of devices failed");

    cl_device_id *dids = new cl_device_id[*num_devices];
    status = clGetDeviceIDs(pid, dev_type, *num_devices, dids, NULL);
    // checkError(status, "Query for device ids");

    return dids;
}

// Searches all platforms for the first platform whose name
// contains the search string (case-insensitive).
cl_platform_id CloverChunk::findPlatform(const char *platform_name_search) {
    cl_int status;

    std::string search = platform_name_search;
    std::transform(search.begin(), search.end(), search.begin(), tolower);

    // Get number of platforms.
    cl_uint num_platforms;
    status = clGetPlatformIDs(0, NULL, &num_platforms);
    // checkError(status, "Query for number of platforms failed");

    // Get a list of all platform ids.
    scoped_array<cl_platform_id> pids(num_platforms);
    status = clGetPlatformIDs(num_platforms, pids, NULL);
    // checkError(status, "Query for all platform ids failed");

    // For each platform, get name and compare against the search string.
    for(unsigned i = 0; i < num_platforms; ++i) {
        std::string name = getPlatformName(pids[i]);

        // Convert to lower case.
        std::transform(name.begin(), name.end(), name.begin(), tolower);

        if(name.find(search) != std::string::npos) {
            // Found!
            return pids[i];
        }
    }

    // No platform found.
    return NULL;
}
