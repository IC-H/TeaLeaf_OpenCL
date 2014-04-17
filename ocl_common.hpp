#ifndef __CL_COMMON_HDR
#define __CL_COMMON_HDR

#include "CL/cl.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>

// 2 dimensional arrays - use a 2D tile for local group
const static size_t LOCAL_X = 128;

#ifdef ONED_KERNEL_LAUNCHES
const static size_t LOCAL_Y = 1;
const static cl::NDRange local_group_size(LOCAL_X);
#else
const static size_t LOCAL_Y = 1;
const static cl::NDRange local_group_size(LOCAL_X, LOCAL_Y);
#endif

// used in update_halo and for copying back to host for mpi transfers
#define FIELD_density0      1
#define FIELD_density1      2
#define FIELD_energy0       3
#define FIELD_energy1       4
#define FIELD_pressure      5
#define FIELD_viscosity     6
#define FIELD_soundspeed    7
#define FIELD_xvel0         8
#define FIELD_xvel1         9
#define FIELD_yvel0         10
#define FIELD_yvel1         11
#define FIELD_vol_flux_x    12
#define FIELD_vol_flux_y    13
#define FIELD_mass_flux_x   14
#define FIELD_mass_flux_y   15
#define FIELD_u             16
#define FIELD_p             17
#define NUM_FIELDS          17
#define FIELD_work_array_1 FIELD_p

// which side to pack - keep the same as in fortran file
#define CHUNK_LEFT 1
#define CHUNK_left 1
#define CHUNK_RIGHT 2
#define CHUNK_right 2
#define CHUNK_BOTTOM 3
#define CHUNK_bottom 3
#define CHUNK_TOP 4
#define CHUNK_top 4
#define EXTERNAL_FACE       (-1)

#define CELL_DATA   1
#define VERTEX_DATA 2
#define X_FACE_DATA 3
#define Y_FACE_DATA 4

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
    cl::Kernel kernel;
    cl::NDRange global_size;
    cl::NDRange local_size;
} reduce_kernel_info_t;

// vectors of kernels and work group sizes for a specific reduction
typedef std::vector<reduce_kernel_info_t> reduce_info_vec_t;

class CloverChunk
{
private:
    // kernels
    cl::Kernel ideal_gas_device;
    cl::Kernel accelerate_device;
    cl::Kernel flux_calc_y_device;
    cl::Kernel flux_calc_x_device;
    cl::Kernel viscosity_device;
    cl::Kernel revert_device;
    cl::Kernel reset_field_device;
    cl::Kernel set_field_device;
    cl::Kernel field_summary_device;
    cl::Kernel calc_dt_device;

    cl::Kernel generate_chunk_device;
    cl::Kernel generate_chunk_init_device;

    cl::Kernel initialise_chunk_first_device;
    cl::Kernel initialise_chunk_second_device;

    cl::Kernel PdV_predict_device;
    cl::Kernel PdV_not_predict_device;

    cl::Kernel advec_mom_vol_device;
    cl::Kernel advec_mom_node_flux_post_x_1_device;
    cl::Kernel advec_mom_node_flux_post_x_2_device;
    cl::Kernel advec_mom_node_pre_x_device;
    cl::Kernel advec_mom_flux_x_device;
    cl::Kernel advec_mom_xvel_device;
    cl::Kernel advec_mom_node_flux_post_y_1_device;
    cl::Kernel advec_mom_node_flux_post_y_2_device;
    cl::Kernel advec_mom_node_pre_y_device;
    cl::Kernel advec_mom_flux_y_device;
    cl::Kernel advec_mom_yvel_device;

    cl::Kernel advec_cell_pre_vol_x_device;
    cl::Kernel advec_cell_ener_flux_x_device;
    cl::Kernel advec_cell_x_device;
    cl::Kernel advec_cell_pre_vol_y_device;
    cl::Kernel advec_cell_ener_flux_y_device;
    cl::Kernel advec_cell_y_device;

    // halo updates
    cl::Kernel update_halo_top_device;
    cl::Kernel update_halo_bottom_device;
    cl::Kernel update_halo_left_device;
    cl::Kernel update_halo_right_device;

    #define TEA_ENUM_JACOBI     1
    #define TEA_ENUM_CG         2
    #define TEA_ENUM_CHEBYSHEV  3
    int tea_solver;

    // TODO could be used by all - precalculate diagonal + scale Kx/Ky
    cl::Kernel tea_leaf_init_diag_device;
    // diagonal of matrix
    cl::Buffer diag;

    // tea leaf
    cl::Kernel tea_leaf_cg_init_u_device;
    cl::Kernel tea_leaf_cg_init_directions_device;
    cl::Kernel tea_leaf_cg_init_others_device;
    cl::Kernel tea_leaf_cg_solve_calc_w_device;
    cl::Kernel tea_leaf_cg_solve_calc_ur_device;
    cl::Kernel tea_leaf_cg_solve_calc_p_device;
    cl::Buffer z;

    // chebyshev solver
    cl::Kernel tea_leaf_cheby_solve_init_p_device;
    cl::Kernel tea_leaf_cheby_solve_calc_u_device;
    cl::Kernel tea_leaf_cheby_solve_calc_p_device;
    cl::Kernel tea_leaf_cheby_solve_calc_resid_device;
    cl::Kernel tea_leaf_cheby_solve_loop_calc_u_device;

    // used to hold the alphas/beta used in chebyshev solver - different from CG ones!
    cl::Buffer ch_alphas_device, ch_betas_device;

    // need more for the Kx/Ky arrays
    cl::Kernel tea_leaf_jacobi_init_device;
    cl::Kernel tea_leaf_jacobi_copy_u_device;
    cl::Kernel tea_leaf_jacobi_solve_device;

    cl::Buffer u;
    cl::Kernel tea_leaf_finalise_device;

    // tolerance specified in tea.in
    float tolerance;

    // calculate rx/ry to pass back to fortran
    void calcrxry
    (double dt, double * rx, double * ry);

    // specific sizes and launch offsets for different kernels
    typedef struct {
        cl::NDRange global;
        cl::NDRange offset;
    } launch_specs_t;
    std::map< std::string, launch_specs_t > launch_specs;

    // reduction kernels - need multiple levels
    reduce_info_vec_t min_red_kernels_double;
    reduce_info_vec_t max_red_kernels_double;
    reduce_info_vec_t sum_red_kernels_double;
    // for PdV
    reduce_info_vec_t max_red_kernels_int;

    // ocl things
    cl::CommandQueue queue;
    cl::Platform platform;
    cl::Device device;
    cl::Context context;

    // for passing into kernels for changing operation based on device type
    std::string device_type_prepro;

    // buffers
    cl::Buffer density0;
    cl::Buffer density1;
    cl::Buffer energy0;
    cl::Buffer energy1;
    cl::Buffer pressure;
    cl::Buffer soundspeed;
    cl::Buffer viscosity;
    cl::Buffer volume;

    cl::Buffer vol_flux_x;
    cl::Buffer vol_flux_y;
    cl::Buffer mass_flux_x;
    cl::Buffer mass_flux_y;

    cl::Buffer cellx;
    cl::Buffer celly;
    cl::Buffer celldx;
    cl::Buffer celldy;
    cl::Buffer vertexx;
    cl::Buffer vertexy;
    cl::Buffer vertexdx;
    cl::Buffer vertexdy;

    cl::Buffer xarea;
    cl::Buffer yarea;

    cl::Buffer xvel0;
    cl::Buffer xvel1;
    cl::Buffer yvel0;
    cl::Buffer yvel1;

    // generic work arrays
    cl::Buffer work_array_1;
    cl::Buffer work_array_2;
    cl::Buffer work_array_3;
    cl::Buffer work_array_4;
    cl::Buffer work_array_5;
    cl::Buffer work_array_6;

    // for reduction in PdV
    cl::Buffer PdV_reduce_buf;

    // for reduction in field_summary
    cl::Buffer reduce_buf_1;
    cl::Buffer reduce_buf_2;
    cl::Buffer reduce_buf_3;
    cl::Buffer reduce_buf_4;
    cl::Buffer reduce_buf_5;
    cl::Buffer reduce_buf_6;

    // global size for kernels
    cl::NDRange global_size;
    // total number of cells
    size_t total_cells;
    // number of cells reduced
    size_t reduced_cells;

    // sizes for launching update halo kernels - l/r and u/d updates
    cl::NDRange update_lr_global_size[2];
    cl::NDRange update_ud_global_size[2];
    cl::NDRange update_lr_local_size[2];
    cl::NDRange update_ud_local_size[2];

    // values used to control operation
    size_t x_min;
    size_t x_max;
    size_t y_min;
    size_t y_max;
    // mpi rank
    int rank;

    // desired type for opencl
    int desired_type;

    // if profiling
    int profiler_on;
    // for recording times if profiling is on
    std::map<std::string, double> kernel_times;

    // Where to send debug output
    FILE* DBGOUT;

    // type of callback for buffer packing
    typedef cl_int (cl::CommandQueue::*buffer_func_t)
    (
        const cl::Buffer&,
        cl_bool,
        const cl::size_t<3>&,
        const cl::size_t<3>&,
        const cl::size_t<3>&,
        size_t,
        size_t,
        size_t,
        size_t,
        void *,
        const std::vector<cl::Event> *,
        cl::Event
    ) const;

    // compile a file and the contained kernels, and check for errors
    void compileKernel
    (const std::string& options,
     const std::string& source_name,
     const char* kernel_name,
     cl::Kernel& kernel);
    cl::Program compileProgram
    (const std::string& source,
     const std::string& options);
    // keep track of built programs to avoid rebuilding them
    std::map<std::string, cl::Program> built_programs;
    std::vector<double> dumpArray
    (const std::string& arr_name, int x_extra, int y_extra);
    std::map<std::string, cl::Buffer> arr_names;

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
    void cloverDie
    (int line, const char* filename, const char* format, ...);

public:
    // kernels
    void calc_dt_kernel(double g_small, double g_big, double dtmin,
        double dtc_safe, double dtu_safe, double dtv_safe,
        double dtdiv_safe, double* dt_min_val, int* dtl_control,
        double* xl_pos, double* yl_pos, int* jldt, int* kldt, int* small);

    void field_summary_kernel(double* vol, double* mass,
        double* ie, double* ke, double* press, double* temp);

    void PdV_kernel(int* error_condition, int predict, double dbyt);

    void ideal_gas_kernel(int predict);

    void generate_chunk_kernel(const int number_of_states, 
        const double* state_density, const double* state_energy,
        const double* state_xvel, const double* state_yvel,
        const double* state_xmin, const double* state_xmax,
        const double* state_ymin, const double* state_ymax,
        const double* state_radius, const int* state_geometry,
        const int g_rect, const int g_circ, const int g_point);

    void initialise_chunk_kernel(double d_xmin, double d_ymin,
        double d_dx, double d_dy);

    void update_halo_kernel(const int* fields, int depth,
        const int* chunk_neighbours);
    void update_array
    (cl::Buffer& cur_array,
    const cell_info_t& array_type,
    const int* chunk_neighbours,
    int depth);

    void accelerate_kernel(double dbyt);

    void advec_mom_kernel(int which_vel, int sweep_number, int direction);

    void flux_calc_kernel(double dbyt);

    void advec_cell_kernel(int dr, int swp_nmbr);

    void revert_kernel();

    void set_field_kernel();
    void reset_field_kernel();

    void viscosity_kernel();

    // Tea leaf
    void tea_leaf_init_jacobi(int, double, double*, double*);
    void tea_leaf_kernel_jacobi(double, double, double*);

    void tea_leaf_init_cg(int, double, double*, double*, double*);
    void tea_leaf_kernel_cg_calc_w(double rx, double ry, double* pw);
    void tea_leaf_kernel_cg_calc_ur(double alpha, double* rrn);
    void tea_leaf_kernel_cg_calc_p(double beta);

    void tea_leaf_cheby_copy_u(void);
    void tea_leaf_calc_2norm_kernel
    (int* initial, double* norm);
    void tea_leaf_kernel_cheby_init
    (const double* rx, const double* ry, const double* theta);
    void tea_leaf_kernel_cheby_iterate
    (const double * ch_alphas, const double * ch_betas, int * n_coefs,
     const double * rx, const double * ry, const int * cheby_calc_steps);

    void tea_leaf_finalise();

    // ctor
    CloverChunk
    (void);
    CloverChunk
    (int* in_x_min, int* in_x_max,
     int* in_y_min, int* in_y_max,
     int* in_profiler_on);
    // dtor
    ~CloverChunk
    (void);

    // enqueue a kernel
    void enqueueKernel
    (cl::Kernel const& kernel,
     int line, const char* file,
     const cl::NDRange offset,
     const cl::NDRange global_range,
     const cl::NDRange local_range,
     const std::vector< cl::Event > * const events=NULL,
     cl::Event * const event=NULL) ;

#if 0
    #define ENQUEUE_OFFSET(knl) ENQUEUE(knl)
#else
    #define ENQUEUE_OFFSET(knl)                                     \
        CloverChunk::enqueueKernel(knl, __LINE__, __FILE__,         \
                                   launch_specs.at(#knl).offset,    \
                                   launch_specs.at(#knl).global,    \
                                   local_group_size);
#endif

    #define ENQUEUE(knl)                                    \
        CloverChunk::enqueueKernel(knl, __LINE__, __FILE__, \
                                   cl::NullRange,           \
                                   global_size,             \
                                   local_group_size);

    // reduction
    template <typename T>
    T reduceValue
    (reduce_info_vec_t& red_kernels,
     const cl::Buffer& results_buf);

    // mpi packing
    #define PACK_ARGS                                       \
        int chunk_1, int chunk_2, int external_face,        \
        int x_inc, int y_inc, int depth, int which_field,   \
        double *buffer_1, double *buffer_2

    void pack_left_right(PACK_ARGS);
    void unpack_left_right(PACK_ARGS);
    void pack_top_bottom(PACK_ARGS);
    void unpack_top_bottom(PACK_ARGS);

    void packRect
    (double* device_buffer, buffer_func_t buffer_func,
     int x_inc, int y_inc, int edge, int dest,
     int which_field, int depth);
};

class KernelCompileError : std::exception
{
private:
    const std::string _err;
public:
    KernelCompileError(const char* err):_err(err){}
    ~KernelCompileError() throw(){}
    const char* what() const throw() {return this->_err.c_str();}
};

extern "C" void tqli_(double *d, double *e, int *np, double **z, int* info);

#endif
