#include "../ctx_common.hpp"
#include "opencl_reduction.hpp"

template <>
void TeaOpenCLTile::getKxKy
<cl::Buffer>
(cl::Buffer * Kx, cl::Buffer * Ky)
{
    Kx = &vector_Kx;
    Ky = &vector_Ky;
}

void TeaOpenCLTile::tea_leaf_dpcg_coarsen_matrix_kernel
(double * host_Kx, double * host_Ky)
{
    ENQUEUE(tea_leaf_dpcg_coarsen_matrix_device);

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_Kx, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        host_Kx);

    queue.enqueueReadBuffer(coarse_local_Ky, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        host_Ky);
}

void TeaOpenCLTile::tea_leaf_dpcg_copy_reduced_coarse_grid
(double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di)
{
    cl::size_t<3> buffer_origin;
    cl::size_t<3> host_origin;
    cl::size_t<3> region;

    // copying from the host, needs to take halos into account
    host_origin[0] = run_params.halo_exchange_depth;
    host_origin[1] = run_params.halo_exchange_depth;
    host_origin[2] = 0;

    buffer_origin[0] = run_params.halo_exchange_depth;
    buffer_origin[1] = run_params.halo_exchange_depth;
    buffer_origin[2] = 0;

    region[0] = tile_x_cells;
    region[1] = tile_y_cells;
    region[2] = 1;

    // convert to bytes
    host_origin[0] *= sizeof(double);
    buffer_origin[0] *= sizeof(double);
    region[0] *= sizeof(double);

    size_t buffer_row_pitch = (tile_x_cells + 2*run_params.halo_exchange_depth)*sizeof(double);
    size_t host_row_pitch = (tile_x_cells + 2*run_params.halo_exchange_depth)*sizeof(double);

    // Need to copy back into the middle of the grid, not in the halos
    queue.enqueueWriteBufferRect(vector_Kx, CL_TRUE,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        0,
        host_row_pitch,
        0,
        global_coarse_Kx);
}

void TeaOpenCLTile::tea_leaf_dpcg_prolong_z_kernel
(double * t2_local)
{
    queue.enqueueWriteBuffer(coarse_local_t2, CL_TRUE, 0, 
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        t2_local);

    ENQUEUE(tea_leaf_dpcg_prolong_Z_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_subtract_u_kernel
(double * t2_local)
{
    queue.enqueueWriteBuffer(coarse_local_t2, CL_TRUE, 0, 
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        t2_local);

    ENQUEUE(tea_leaf_dpcg_subtract_u_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_restrict_zt_kernel
(double * ztr_local)
{
    ENQUEUE(tea_leaf_dpcg_restrict_ZT_device);

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_ztr, CL_TRUE, 0, 
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        ztr_local);
}

void TeaOpenCLTile::tea_leaf_dpcg_matmul_zta_kernel
(double * ztaz_local)
{
    ENQUEUE(tea_leaf_dpcg_matmul_ZTA_device);

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_ztaz, CL_TRUE, 0, 
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        ztaz_local);
}

void TeaOpenCLTile::tea_leaf_dpcg_init_p_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_init_p_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_store_r_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_store_r_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_calc_rrn_kernel
(double * rrn)
{
    //ENQUEUE(tea_leaf_dpcg_calc_rrn_device);

    //*rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
}

void TeaOpenCLTile::tea_leaf_dpcg_calc_p_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_calc_p_device);
}

