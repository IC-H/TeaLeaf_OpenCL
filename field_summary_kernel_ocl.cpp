#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void field_summary_kernel_ocl_
(double* vol, double* mass, double* ie, double* temp)
{
    tea_context.field_summary_kernel(vol, mass, ie, temp);
}

void TeaCLContext::field_summary_kernel
(double* vol, double* mass, double* ie, double* temp)
{
    FOR_EACH_TILE
    {
        ENQUEUE(field_summary_device);

        *vol = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_1);
        *mass = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_2);
        *ie = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_3);
        *temp = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_4);
    }
}

