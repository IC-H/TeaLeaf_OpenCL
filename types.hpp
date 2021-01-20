#ifndef __CL_TYPE_HDR
#define __CL_TYPE_HDR

#include <cstdio>
#include <cstdlib>
#include <map>

typedef struct cell_info {
    const int x_extra;
    const int y_extra;
    const int x_invert;
    const int y_invert;
    const int x_face;
    const int y_face;
    const int grid_type;

    cell_info
    (int in_x_extra, int in_y_extra,
    int in_x_invert, int in_y_invert,
    int in_x_face, int in_y_face,
    int in_grid_type)
    :x_extra(in_x_extra), y_extra(in_y_extra),
    x_invert(in_x_invert), y_invert(in_y_invert),
    x_face(in_x_face), y_face(in_y_face),
    grid_type(in_grid_type)
    {
        ;
    }

} cell_info_t;

// reductions
typedef struct red_t {
    cl_kernel kernel;
    cl::NDRange global_size;
    cl::NDRange local_size;
} reduce_kernel_info_t;

// vectors of kernels and work group sizes for a specific reduction
typedef std::vector<reduce_kernel_info_t> reduce_info_vec_t;

class CloverChunk
{
private:
    // kernels
    cl_kernel set_field_device;
    cl_kernel field_summary_device;

    cl_kernel generate_chunk_device;
    cl_kernel generate_chunk_init_device;
    cl_kernel generate_chunk_init_u_device;

    cl_kernel initialise_chunk_first_device;
    cl_kernel initialise_chunk_second_device;

    // halo updates
    cl_kernel update_halo_top_device;
    cl_kernel update_halo_bottom_device;
    cl_kernel update_halo_left_device;
    cl_kernel update_halo_right_device;
    // mpi packing
    cl_kernel pack_left_buffer_device;
    cl_kernel unpack_left_buffer_device;
    cl_kernel pack_right_buffer_device;
    cl_kernel unpack_right_buffer_device;
    cl_kernel pack_bottom_buffer_device;
    cl_kernel unpack_bottom_buffer_device;
    cl_kernel pack_top_buffer_device;
    cl_kernel unpack_top_buffer_device;

    // main buffers, with sub buffers for each offset
    cl_mem left_buffer;
    cl_mem right_buffer;
    cl_mem bottom_buffer;
    cl_mem top_buffer;

    #define TEA_ENUM_JACOBI     1
    #define TEA_ENUM_CG         2
    #define TEA_ENUM_CHEBYSHEV  3
    #define TEA_ENUM_PPCG       4
    int tea_solver;

    // tea leaf
    cl_kernel tea_leaf_cg_solve_init_p_device;
    cl_kernel tea_leaf_cg_solve_calc_w_device;
    cl_kernel tea_leaf_cg_solve_calc_ur_device;
    cl_kernel tea_leaf_cg_solve_calc_rrn_device;
    cl_kernel tea_leaf_cg_solve_calc_p_device;
    cl_mem vector_z;

    // chebyshev solver
    cl_kernel tea_leaf_cheby_solve_init_p_device;
    cl_kernel tea_leaf_cheby_solve_calc_u_device;
    cl_kernel tea_leaf_cheby_solve_calc_p_device;
    cl_kernel tea_leaf_calc_2norm_device;

    cl_kernel tea_leaf_ppcg_solve_init_sd_device;
    cl_kernel tea_leaf_ppcg_solve_calc_sd_device;
    cl_kernel tea_leaf_ppcg_solve_update_r_device;

    // used to hold the alphas/beta used in chebyshev solver - different from CG ones!
    cl_mem ch_alphas_device, ch_betas_device;

    // need more for the Kx/Ky arrays
    cl_kernel tea_leaf_jacobi_copy_u_device;
    cl_kernel tea_leaf_jacobi_solve_device;

    cl_kernel tea_leaf_block_init_device;
    cl_kernel tea_leaf_block_solve_device;
    cl_kernel tea_leaf_init_jac_diag_device;
    cl_mem cp, bfp;

    cl_mem u, u0;
    cl_kernel tea_leaf_finalise_device;
    // TODO could be used by all - precalculate diagonal + scale Kx/Ky
    cl_kernel tea_leaf_calc_residual_device;
    cl_kernel tea_leaf_init_common_device;
    cl_kernel tea_leaf_zero_boundary_device;

    // tolerance specified in tea.in
    float tolerance;
    // type of preconditioner
    int preconditioner_type;

    // calculate rx/ry to pass back to fortran
    void calcrxry
    (double dt, double * rx, double * ry);

    // specific sizes and launch offsets for different kernels
    typedef struct {
        cl::NDRange global;
        cl::NDRange offset;
    } launch_specs_t;
    std::map< std::string, launch_specs_t > launch_specs;

    launch_specs_t findPaddingSize
    (int vmin, int vmax, int hmin, int hmax);

    // reduction kernels - need multiple levels
    reduce_info_vec_t min_red_kernels_double;
    reduce_info_vec_t max_red_kernels_double;
    reduce_info_vec_t sum_red_kernels_double;
    // for PdV
    reduce_info_vec_t max_red_kernels_int;

    // ocl things
    cl_command_queue queue;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_int status;

    // for passing into kernels for changing operation based on device type
    std::string device_type_prepro;

    // buffers
    cl_mem density;
    cl_mem energy0;
    cl_mem energy1;
    cl_mem volume;

    cl_mem cellx;
    cl_mem celly;
    cl_mem celldx;
    cl_mem celldy;
    cl_mem vertexx;
    cl_mem vertexy;
    cl_mem vertexdx;
    cl_mem vertexdy;

    cl_mem xarea;
    cl_mem yarea;

    // generic work arrays
    cl_mem vector_p;
    cl_mem vector_r;
    cl_mem vector_w;
    cl_mem vector_Mi;
    cl_mem vector_Kx;
    cl_mem vector_Ky;
    cl_mem vector_sd;

    // for reduction in PdV
    cl_mem PdV_reduce_buf;

    // for reduction in field_summary
    cl_mem reduce_buf_1;
    cl_mem reduce_buf_2;
    cl_mem reduce_buf_3;
    cl_mem reduce_buf_4;
    cl_mem reduce_buf_5;
    cl_mem reduce_buf_6;

    // global size for kernels
    cl::NDRange global_size;
    // number of cells reduced
    int reduced_cells;

    // halo size
    int halo_exchange_depth;

    // sizes for launching update halo kernels - l/r and u/d updates
    std::map<int, cl::NDRange> update_lr_global_size;
    std::map<int, cl::NDRange> update_bt_global_size;
    std::map<int, cl::NDRange> update_lr_local_size;
    std::map<int, cl::NDRange> update_bt_local_size;
    std::map<int, cl::NDRange> update_lr_offset;
    std::map<int, cl::NDRange> update_bt_offset;

    // values used to control operation
    int x_min;
    int x_max;
    int y_min;
    int y_max;
    // mpi rank
    int rank;

    // desired type for opencl
    int desired_type;

    // if profiling
    bool profiler_on;
    // for recording times if profiling is on
    std::map<std::string, double> kernel_times;
    // recording number of times each kernel was called
    std::map<std::string, int> kernel_calls;

    // Where to send debug output
    FILE* DBGOUT;

    // compile a file and the contained kernels, and check for errors
    void compileKernel
    (std::stringstream& options,
     const std::string& source_name,
     const char* kernel_name,
     cl_kernel& kernel,
     int launch_x_min, int launch_x_max,
     int launch_y_min, int launch_y_max);
    cl_program compileProgram
    (const std::string& source,
     const std::string& options);
    // keep track of built programs to avoid rebuilding them
    std::map<std::string, cl_program> built_programs;
    std::vector<double> dumpArray
    (const std::string& arr_name, int x_extra, int y_extra);
    std::map<std::string, cl_mem> arr_names;

    /*
     *  initialisation subroutines
     */

    // initialise context, queue, etc
    void initOcl
    (void);
    // initialise all program stuff, kernels, etc
    void initProgram
    (void);
    // intialise local/global sizes
    void initSizes
    (void);
    // initialise buffers for device
    void initBuffers
    (void);
    // initialise all the arguments for each kernel
    void initArgs
    (void);
    // create reduction kernels
    void initReduction
    (void);

    // this function gets called when something goes wrong
    #define DIE(...) cloverDie(__LINE__, __FILE__, __VA_ARGS__)

public:
    void field_summary_kernel(double* vol, double* mass,
        double* ie, double* temp);

    void generate_chunk_kernel(const int number_of_states, 
        const double* state_density, const double* state_energy,
        const double* state_xmin, const double* state_xmax,
        const double* state_ymin, const double* state_ymax,
        const double* state_radius, const int* state_geometry,
        const int g_rect, const int g_circ, const int g_point);

    void initialise_chunk_kernel(double d_xmin, double d_ymin,
        double d_dx, double d_dy);

    void update_halo_kernel(const int* fields, int depth,
        const int* chunk_neighbours);
    void update_array
    (cl_mem& cur_array,
    const cell_info_t& array_type,
    const int* chunk_neighbours,
    int depth);

    void set_field_kernel();

    std::string getPlatformName(cl_platform_id pid);

    cl_device_id *getDevices(cl_platform_id pid, cl_device_type dev_type, cl_uint *num_devices);

    cl_platform_id findPlatform(const char *platform_name_search);

    // Tea leaf
    void tea_leaf_jacobi_solve_kernel
    (double* error);

    void tea_leaf_cg_init_kernel
    (double * rro);
    void tea_leaf_cg_calc_w_kernel
    (double* pw);
    void tea_leaf_cg_calc_ur_kernel
    (double alpha, double* rrn);
    void tea_leaf_cg_calc_p_kernel
    (double beta);

    void tea_leaf_calc_2norm_kernel
    (int norm_array, double* norm);
    void tea_leaf_cheby_init_kernel
    (const double * ch_alphas, const double * ch_betas, int n_coefs,
     const double rx, const double ry, const double theta);
    void tea_leaf_cheby_iterate_kernel
    (const int cheby_calc_steps);

    void ppcg_init
    (const double * ch_alphas, const double * ch_betas,
    const double theta, const int n);
    void ppcg_init_sd_kernel
    (void);
    void tea_leaf_ppcg_inner_kernel
    (int, int, const int*);

    void tea_leaf_finalise();
    void tea_leaf_calc_residual(void);
    void tea_leaf_common_init
    (int coefficient, double dt, double * rx, double * ry,
     int * zero_boundary, int reflective_boundary);

    void print_profiling_info
    (void);

    // ctor
    CloverChunk
    (void);
    CloverChunk
    (int* in_x_min, int* in_x_max,
     int* in_y_min, int* in_y_max);

    // enqueue a kernel
    void enqueueKernel
    (cl_kernel const& kernel,
     int line, const char* file,
     const cl::NDRange offset,
     const cl::NDRange global_range,
     const cl::NDRange local_range,
     const std::vector< cl::Event > * const events=NULL,
     cl::Event * const event=NULL) ;

    #define ENQUEUE_OFFSET(knl)                        \
        enqueueKernel(knl, __LINE__, __FILE__,         \
                      launch_specs.at(#knl).offset,    \
                      launch_specs.at(#knl).global,    \
                      local_group_size);

    // reduction
    template <typename T>
    T reduceValue
    (reduce_info_vec_t& red_kernels,
     const cl_mem& results_buf);

    void packUnpackAllBuffers
    (int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int depth,
     int face, int pack, double * buffer);
};

class KernelCompileError : std::exception
{
private:
    const std::string _what;
    const int _err;
public:
    KernelCompileError(const char* what, int err):_what(what),_err(err){}
    ~KernelCompileError() throw(){}

    const char* what() const throw() {return this->_what.c_str();}

    const int err() const throw() {return this->_err;}
};

#endif
