#ifndef __CL_REDUCTION_HDR
#define __CL_REDUCTION_HDR

#include "ocl_common.hpp"

template <typename T>
T CloverChunk::reduceValue
(reduce_info_vec_t& red_kernels,
 const cl_mem& results_buf)
{
    // enqueue the kernels in order
    for (size_t ii = 0; ii < red_kernels.size(); ii++)
    {
        red_kernels.at(ii).kernel.setArg(0, results_buf);
        CloverChunk::enqueueKernel(red_kernels.at(ii).kernel,
                                   __LINE__, __FILE__,
                                   cl::NullRange,
                                   red_kernels.at(ii).global_size,
                                   red_kernels.at(ii).local_size);
    }

    T result;

    // make sure final reduction has finished
    clFinish(queue);

    // copy back the result and return
    status = clEnqueueReadBuffer(queue, results_buf, CL_TRUE, 0, sizeof(T), &result, 0, NULL, NULL);
    // checkError(status, "Failed to copy expand_to from device");

    return result;
}

#endif

