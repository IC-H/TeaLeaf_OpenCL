#include "../ctx_common.hpp"
#include "opencl_reduction.hpp"

#include <cmath>

// TODO some of these might not have to copy memory back and forth as much as they do

void TeaOpenCLChunk::tea_leaf_dpcg_coarsen_matrix_kernel
(double * host_Kx, double * host_Ky)
{
    ENQUEUE_DEFLATION(tea_leaf_dpcg_coarsen_matrix_device);

    // These need to be run because init_cg isn't called
    if (run_params.preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        ENQUEUE(tea_leaf_block_init_device);
        ENQUEUE(tea_leaf_block_solve_device);
    }
    else if (run_params.preconditioner_type == TL_PREC_JAC_DIAG)
    {
        ENQUEUE(tea_leaf_init_jac_diag_device);
    }

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_Kx, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        host_Kx);

    queue.enqueueReadBuffer(coarse_local_Ky, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        host_Ky);
}

void TeaOpenCLChunk::getCoarseCopyParameters
(cl::size_t<3> * buffer_origin,
 cl::size_t<3> * host_origin,
 cl::size_t<3> * region,
 size_t * buffer_row_pitch,
 size_t * host_row_pitch)
{
    // copying from the host, needs to take halos into account
    (*host_origin)[0] = run_params.halo_exchange_depth;
    (*host_origin)[1] = run_params.halo_exchange_depth;
    (*host_origin)[2] = 0;

    (*buffer_origin)[0] = run_params.halo_exchange_depth;
    (*buffer_origin)[1] = run_params.halo_exchange_depth;
    (*buffer_origin)[2] = 0;

    (*region)[0] = chunk_x_cells;
    (*region)[1] = chunk_y_cells;
    (*region)[2] = 1;

    // convert to bytes
    (*host_origin)[0] *= sizeof(double);
    (*buffer_origin)[0] *= sizeof(double);
    (*region)[0] *= sizeof(double);

    (*buffer_row_pitch) = (chunk_x_cells + 2*run_params.halo_exchange_depth)*sizeof(double);
    (*host_row_pitch) = (chunk_x_cells + 2*run_params.halo_exchange_depth)*sizeof(double);

}

void TeaOpenCLChunk::writeRect
(cl::Buffer dst, double * src)
{
    cl::size_t<3> buffer_origin;
    cl::size_t<3> host_origin;
    cl::size_t<3> region;

    size_t buffer_row_pitch;
    size_t host_row_pitch;

    getCoarseCopyParameters(&buffer_origin, &host_origin, &region,
        &buffer_row_pitch, &host_row_pitch);

    queue.enqueueWriteBufferRect(dst,
        CL_TRUE,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        0,
        host_row_pitch,
        0,
        src);
}

void TeaOpenCLChunk::readRect
(double * dst, cl::Buffer src)
{
    cl::size_t<3> buffer_origin;
    cl::size_t<3> host_origin;
    cl::size_t<3> region;

    size_t buffer_row_pitch;
    size_t host_row_pitch;

    getCoarseCopyParameters(&buffer_origin, &host_origin, &region,
        &buffer_row_pitch, &host_row_pitch);

    queue.enqueueReadBufferRect(src,
        CL_TRUE,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        0,
        host_row_pitch,
        0,
        dst);
}

void TeaOpenCLChunk::tea_leaf_dpcg_copy_reduced_coarse_grid
(double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di)
{
    writeRect(vector_Kx, global_coarse_Kx);
    writeRect(vector_Ky, global_coarse_Ky);

    // matmul needs a scaling factor because the diagonal is the size of the sub tile + the others, not 1.0
    double scale_factor = double(SUB_TILE_BLOCK_SIZE*SUB_TILE_BLOCK_SIZE);

    tea_leaf_cg_solve_calc_w_device.setArg(6, scale_factor);
    tea_leaf_ppcg_solve_update_r_device.setArg(8, scale_factor);
    tea_leaf_calc_residual_device.setArg(6, scale_factor);
    tea_leaf_init_jac_diag_device.setArg(4, scale_factor);
    tea_leaf_dpcg_matmul_ZTA_device.setArg(5, scale_factor);
}

void TeaOpenCLChunk::tea_leaf_dpcg_prolong_z_kernel
(double * t2_local)
{
    queue.enqueueWriteBuffer(coarse_local_t2, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        t2_local);

    ENQUEUE_DEFLATION(tea_leaf_dpcg_prolong_Z_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_subtract_u_kernel
(double * t2_local)
{
    queue.enqueueWriteBuffer(coarse_local_t2, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        t2_local);

    ENQUEUE_DEFLATION(tea_leaf_dpcg_subtract_u_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_restrict_zt_kernel
(double * ztr_local)
{
    ENQUEUE_DEFLATION(tea_leaf_dpcg_restrict_ZT_device);

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_ztr, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        ztr_local);
}

void TeaOpenCLChunk::tea_leaf_dpcg_copy_reduced_t2
(double * global_coarse_t2)
{
    writeRect(u0, global_coarse_t2);
}

void TeaOpenCLChunk::tea_leaf_dpcg_solve_z
(void)
{
    ENQUEUE(tea_leaf_dpcg_solve_z_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_matmul_zta_kernel
(double * ztaz_local)
{
    ENQUEUE_DEFLATION(tea_leaf_dpcg_matmul_ZTA_device);

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_ztaz, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        ztaz_local);
}

void TeaOpenCLChunk::tea_leaf_dpcg_init_p_kernel
(void)
{
    ENQUEUE(tea_leaf_dpcg_init_p_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_store_r_kernel
(void)
{
    ENQUEUE(tea_leaf_dpcg_store_r_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_calc_rrn_kernel
(double * rrn)
{
    ENQUEUE(tea_leaf_dpcg_calc_rrn_device);

    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
}

void TeaOpenCLChunk::tea_leaf_dpcg_calc_p_kernel
(double beta)
{
    tea_leaf_dpcg_calc_p_device.setArg(3, beta);

    ENQUEUE(tea_leaf_dpcg_calc_p_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_local_solve
(double   coarse_solve_eps,
 int      coarse_solve_max_iters,
 int    * it_count,
 double   theta,
 int      inner_use_ppcg,
 int      ppcg_max_iters,
 double * inner_cg_alphas,
 double * inner_cg_betas,
 double * inner_ch_alphas,
 double * inner_ch_betas,
 double * t2_result)
{
    writeRect(u, t2_result);

    double rro, rrn, pw;

    tea_leaf_calc_residual();
    tea_leaf_cg_init_kernel(&rro);

    double initial = rro;

    rrn = 1e10;

    //fprintf(stdout, "initial: %+.15e\n", initial);

    if (inner_use_ppcg)
    {
        // FIXME only needs to be done once
        ppcg_init(inner_ch_alphas, inner_ch_betas, theta, ppcg_max_iters);
    }

    for (int ii = 0; (ii < coarse_solve_max_iters) && (sqrt(fabs(rrn)) > coarse_solve_eps*initial); ii++)
    {
        // TODO redo these so it doesnt copy back memory repeatedly
        tea_leaf_cg_calc_w_kernel(&pw);

        double alpha = rro/pw;

        tea_leaf_cg_calc_ur_kernel(alpha, &rrn);

        if (inner_use_ppcg)
        {
            ppcg_init_sd_kernel();

            for (int jj = 0; jj < 10; jj++)
            {
                int zeros[4] = {EXTERNAL_FACE, EXTERNAL_FACE, EXTERNAL_FACE, EXTERNAL_FACE};
                tea_leaf_ppcg_inner_kernel(jj + 1, run_params.halo_exchange_depth, zeros);
            }

            tea_leaf_calc_2norm_kernel(2, &rrn);
        }

        double beta = rrn/rro;

        tea_leaf_cg_calc_p_kernel(beta);

        rro = rrn;

        inner_cg_alphas[ii] = alpha;
        inner_cg_betas[ii] = beta;

        *it_count = ii + 1;
    }

    //fprintf(stdout, "after: %e\n", rrn);
    //fprintf(stdout, "%d iters\n", *it_count);
    //fprintf(stdout, "\n");

    readRect(t2_result, u);
}

