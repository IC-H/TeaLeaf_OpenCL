#include "ocl_common.hpp"

extern "C" void initialise_chunk_kernel_ocl_
(double* d_xmin, double* d_ymin, double* d_dx, double* d_dy)
{
    tea_context.initialise_chunk_kernel(*d_xmin, *d_ymin, *d_dx, *d_dy);
}

void TeaCLContext::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{
    FOR_EACH_TILE
    {
        double tile_d_xmin = d_xmin + (d_dx*tile->left);
        double tile_d_ymin = d_ymin + (d_dy*tile->bottom);

        tile->launch_specs.at("initialise_chunk_first_device").offset = cl::NullRange;
        tile->initialise_chunk_first_device.setArg(0, tile_d_xmin);
        tile->initialise_chunk_first_device.setArg(1, tile_d_ymin);
        tile->initialise_chunk_first_device.setArg(2, d_dx);
        tile->initialise_chunk_first_device.setArg(3, d_dy);

        tile->initialise_chunk_second_device.setArg(0, tile_d_xmin);
        tile->initialise_chunk_second_device.setArg(1, tile_d_ymin);
        tile->initialise_chunk_second_device.setArg(2, d_dx);
        tile->initialise_chunk_second_device.setArg(3, d_dy);

        ENQUEUE(initialise_chunk_first_device);
        ENQUEUE(initialise_chunk_second_device);
    }
}

