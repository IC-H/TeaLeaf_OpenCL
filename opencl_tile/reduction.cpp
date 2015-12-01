#include "../ocl_common.hpp"

void TeaOpenCLTile::initReduction
(void)
{
    /*
     *  create a reduction kernel, one for each layer, with the right parameters
     */

    /*
     *  all the reductions only operate on the inner cells, because the halo
     *  cells aren't really part of the simulation. create a new global size
     *  that doesn't include these halo cells for the reduction which should
     *  speed it up a bit
     */
    const int red_x = tile_x_cells +
        (((tile_x_cells)%LOCAL_X == 0) ? 0 : (LOCAL_X - ((tile_x_cells)%LOCAL_X)));
    const int red_y = tile_y_cells +
        (((tile_y_cells)%LOCAL_Y == 0) ? 0 : (LOCAL_Y - ((tile_y_cells)%LOCAL_Y)));
    reduced_cells = red_x*red_y;

    // each work group reduces to 1 value inside each kernel
    const int total_to_reduce = ceil(float(reduced_cells)/(LOCAL_X*LOCAL_Y));

    fprintf(DBGOUT, "Total cells to reduce = %d\n", reduced_cells);
    fprintf(DBGOUT, "Reduction within work group reduces to = %d\n", total_to_reduce);

    int ii = 0;

    // amount of elements to reduce in serial in OpenCL for good coalescence
    // 16 or 32 is good
    #define SERIAL_REDUCTION_AMOUNT 16

    // number of elements to reduce at this step
    int stage_elems_to_reduce = total_to_reduce;

    while (++ii)
    {
        // different kernels for different types and operations
        cl::Kernel sum_double;

        // make options again
        std::stringstream options("");

#ifdef __arm__
        options << "-D CLOVER_NO_BUILTINS ";
#endif

        // which stage this reduction kernel is at - starts at 1
        options << "-D RED_STAGE=" << ii << " ";
        // original total number of elements to reduce
        options << "-D ORIG_ELEMS_TO_REDUCE=" << total_to_reduce << " ";

        // device type in the form "-D..."
        options << device_type_prepro;
        options << "-w ";

        // the actual number of elements that needs to be reduced in this stage
        options << "-D ELEMS_TO_REDUCE=" << stage_elems_to_reduce << " ";

        options << "-D SERIAL_REDUCTION_AMOUNT=" << SERIAL_REDUCTION_AMOUNT << " ";

        // global size at this step
        int reduction_global_size = std::max(SERIAL_REDUCTION_AMOUNT, int(std::ceil(stage_elems_to_reduce/SERIAL_REDUCTION_AMOUNT)));

        fprintf(DBGOUT, "\n\nStage %d:\n", ii);
        fprintf(DBGOUT, "%d elements remaining to reduce\n", stage_elems_to_reduce);
        fprintf(DBGOUT, "Global size %d\n", reduction_global_size);

        /*
         *  To get the local size to use at this stage, figure out the largest
         *  power of 2 that is under the global size
         *
         *  NB at the moment, enforcing power of 2 local size anyway
         *  NB also, 128 was preferred work group size on phi
         */
        int reduction_local_size = LOCAL_X*LOCAL_Y;

        if (reduction_local_size < SERIAL_REDUCTION_AMOUNT)
        {
            DIE("SERIAL_REDUCTION_AMOUNT (%d) should be less than reduction_local_size (%d)", int(SERIAL_REDUCTION_AMOUNT), reduction_local_size);
        }

        // if there are more elements to reduce than the standard local size
        if (reduction_global_size > reduction_local_size)
        {
            /*
             *  If the standard reduction size is smaller than the number of
             *  actual elements remaining, then do a reduction as normal,
             *  writing back multiple values into the reduction buffer
             */

            /*
             *  Calculate the total number of threads to launch at this stage by
             *  making it divisible by the local size so that a binary reduction can
             *  be done, then dividing it by 2 to account for each thread possibly
             *  being able to load 2 values at once
             *
             *  Keep track of original value for use in load threshold calculation
             */
            while (reduction_global_size % reduction_local_size)
            {
                reduction_global_size++;
            }
        }
        else
        {
            /*
             *  If we are down to a number of elements that is less than we can
             *  fit into one workgroup, then just launch one workgroup which
             *  finishes the reduction
             */
            while (reduction_local_size >= reduction_global_size*2)
            {
                reduction_local_size /= 2;
            }

            /*
             *  launch one work group to finish
             */
            reduction_global_size = reduction_local_size;
        }

        fprintf(DBGOUT, "Padded total number of threads to launch is %d\n", reduction_global_size);
        fprintf(DBGOUT, "Local size for reduction is %d\n", reduction_local_size);

        options << "-D GLOBAL_SZ=" << reduction_global_size << " ";
        options << "-D LOCAL_SZ=" << reduction_local_size << " ";

        options << "-I. ";

        fprintf(DBGOUT, "\n");

        // name of reduction kernel, data type, what the reduction does
        #define MAKE_REDUCE_KNL(name, data_type, init_val)          \
        {                                                           \
            options << "-D red_" << #name << " ";                   \
            options << "-D reduce_t="#data_type << " ";             \
            options << "-D INIT_RED_VAL=" << #init_val << " ";      \
            fprintf(DBGOUT, "Making reduction kernel '%s_%s' ",     \
                    #name, #data_type);                             \
            fprintf(DBGOUT, "with options string:\n%s\n",           \
                    options.str().c_str());                         \
            try                                                     \
            {                                                       \
                compileKernel(options,                              \
                    "./kernel_files/reduction_cl.cl",               \
                    "reduction",                                    \
                    name##_##data_type, 0, 0, 0, 0);                \
            }                                                       \
            catch (KernelCompileError err)                          \
            {                                                       \
                DIE("Errors in compiling reduction %s_%s:\n%s\n",   \
                    #name, #data_type, err.what());                 \
            }                                                       \
            fprintf(DBGOUT, "Kernel '%s_%s' successfully built\n",  \
                    #name, #data_type);                             \
            reduce_kernel_info_t info;                              \
            info.kernel = name##_##data_type;                       \
            info.global_size = cl::NDRange(reduction_global_size);  \
            info.local_size = cl::NDRange(reduction_local_size);    \
            name##_red_kernels_##data_type.push_back(info);         \
            fprintf(DBGOUT, "\n");                                  \
        }

        MAKE_REDUCE_KNL(sum, double, 0.0);

        fprintf(DBGOUT, "%d/%d", stage_elems_to_reduce, (SERIAL_REDUCTION_AMOUNT*reduction_local_size));
        stage_elems_to_reduce = std::ceil((1.0*stage_elems_to_reduce)/(SERIAL_REDUCTION_AMOUNT*reduction_local_size));
        fprintf(DBGOUT, " = %d remaining\n", stage_elems_to_reduce);

        if (stage_elems_to_reduce <= 1)
        {
            break;
        }
    }
}