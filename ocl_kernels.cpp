#include "ocl_common.hpp"
#include <sstream>
#include <fstream>

void CloverChunk::initProgram
(void)
{
    // options
    std::stringstream options("");

#ifdef __arm__
    // on ARM, don't use built in functions as they don't exist
    options << "-DCLOVER_NO_BUILTINS ";
#endif

    options << "-DPRECONDITIONER=" << preconditioner_type << " ";

    // pass in these values so you don't have to pass them in to every kernel
    options << "-Dx_max=" << x_max << " ";
    options << "-Dy_max=" << y_max << " ";

    options << "-DJACOBI_BLOCK_SIZE=" << JACOBI_BLOCK_SIZE << " ";

    // if it doesn't subdivide exactly, need to make sure it doesn't go off the edge
    // rather expensive check so don't always do it
    if (y_max % JACOBI_BLOCK_SIZE)
    {
        options << "-DBLOCK_TOP_CHECK ";
    }

    // local sizes
    options << "-DBLOCK_SZ=" << LOCAL_X*LOCAL_Y << " ";

    // include current directory
    options << "-I. ";

    // device type in the form "-D..."
    options << device_type_prepro;

    // depth of halo in terms of memory allocated, NOT in terms of the actual halo size (which might be different)
    options << "-DHALO_DEPTH=" << halo_exchange_depth << " ";

    if (!rank)
    {
        fprintf(DBGOUT, "Compiling kernels with options:\n%s\n", options.str().c_str());
        fprintf(stdout, "Compiling kernels (may take some time)...");
        fflush(stdout);
    }

    // launch with special work group sizes to cover the whole grid
    compileKernel(options, "./kernel_files/initialise_chunk_cl.cl", "initialise_chunk_first", initialise_chunk_first_device, -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);

    compileKernel(options, "./kernel_files/initialise_chunk_cl.cl", "initialise_chunk_second", initialise_chunk_second_device, -1, 1, -1, 1);
    compileKernel(options, "./kernel_files/generate_chunk_cl.cl", "generate_chunk_init", generate_chunk_init_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/generate_chunk_cl.cl", "generate_chunk_init_u", generate_chunk_init_u_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/generate_chunk_cl.cl", "generate_chunk", generate_chunk_device, 0, 0, 0, 0);

    compileKernel(options, "./kernel_files/set_field_cl.cl", "set_field", set_field_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/field_summary_cl.cl", "field_summary", field_summary_device, 0, 0, 0, 0);

    compileKernel(options, "./kernel_files/update_halo_cl.cl", "update_halo_top", update_halo_top_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/update_halo_cl.cl", "update_halo_bottom", update_halo_bottom_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/update_halo_cl.cl", "update_halo_left", update_halo_left_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/update_halo_cl.cl", "update_halo_right", update_halo_right_device, 0, 0, 0, 0);

    compileKernel(options, "./kernel_files/pack_kernel_cl.cl", "pack_left_buffer", pack_left_buffer_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/pack_kernel_cl.cl", "unpack_left_buffer", unpack_left_buffer_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/pack_kernel_cl.cl", "pack_right_buffer", pack_right_buffer_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/pack_kernel_cl.cl", "unpack_right_buffer", unpack_right_buffer_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/pack_kernel_cl.cl", "pack_bottom_buffer", pack_bottom_buffer_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/pack_kernel_cl.cl", "unpack_bottom_buffer", unpack_bottom_buffer_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/pack_kernel_cl.cl", "pack_top_buffer", pack_top_buffer_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/pack_kernel_cl.cl", "unpack_top_buffer", unpack_top_buffer_device, 0, 0, 0, 0);

    if (tea_solver == TEA_ENUM_CG ||
    tea_solver == TEA_ENUM_CHEBYSHEV ||
    tea_solver == TEA_ENUM_PPCG)
    {
        compileKernel(options, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_solve_calc_w", tea_leaf_cg_solve_calc_w_device, 0, 0, 0, 0);
        compileKernel(options, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_solve_calc_ur", tea_leaf_cg_solve_calc_ur_device, 0, 0, 0, 0);
        compileKernel(options, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_solve_calc_p", tea_leaf_cg_solve_calc_p_device, 0, 0, 0, 0);
        compileKernel(options, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_solve_init_p", tea_leaf_cg_solve_init_p_device, 0, 0, 0, 0);

        if (tea_solver == TEA_ENUM_CHEBYSHEV)
        {
            compileKernel(options, "./kernel_files/tea_leaf_cheby_cl.cl", "tea_leaf_cheby_solve_init_p", tea_leaf_cheby_solve_init_p_device, 0, 0, 0, 0);
            compileKernel(options, "./kernel_files/tea_leaf_cheby_cl.cl", "tea_leaf_cheby_solve_calc_u", tea_leaf_cheby_solve_calc_u_device, 0, 0, 0, 0);
            compileKernel(options, "./kernel_files/tea_leaf_cheby_cl.cl", "tea_leaf_cheby_solve_calc_p", tea_leaf_cheby_solve_calc_p_device, 0, 0, 0, 0);
        }
        else if (tea_solver == TEA_ENUM_PPCG)
        {
            compileKernel(options, "./kernel_files/tea_leaf_ppcg_cl.cl", "tea_leaf_ppcg_solve_init_sd", tea_leaf_ppcg_solve_init_sd_device, 0, 0, 0, 0);
            compileKernel(options, "./kernel_files/tea_leaf_ppcg_cl.cl", "tea_leaf_ppcg_solve_calc_sd", tea_leaf_ppcg_solve_calc_sd_device,
                -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);
            compileKernel(options, "./kernel_files/tea_leaf_ppcg_cl.cl", "tea_leaf_ppcg_solve_update_r", tea_leaf_ppcg_solve_update_r_device, 
                -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);
        }
    }
    else
    {
        compileKernel(options, "./kernel_files/tea_leaf_jacobi_cl.cl", "tea_leaf_jacobi_copy_u", tea_leaf_jacobi_copy_u_device, 0, 0, 0, 0);
        compileKernel(options, "./kernel_files/tea_leaf_jacobi_cl.cl", "tea_leaf_jacobi_solve", tea_leaf_jacobi_solve_device, 0, 0, 0, 0);
    }

    compileKernel(options, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_finalise", tea_leaf_finalise_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_calc_residual", tea_leaf_calc_residual_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_calc_2norm", tea_leaf_calc_2norm_device, 0, 0, 0, 0);

    compileKernel(options, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_block_init", tea_leaf_block_init_device, 0, 0, 0, 0);
    compileKernel(options, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_block_solve", tea_leaf_block_solve_device, 0, 0, 0, 0);

    compileKernel(options, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_init_common", tea_leaf_init_common_device, 1-halo_exchange_depth, halo_exchange_depth, 1-halo_exchange_depth, halo_exchange_depth);
    compileKernel(options, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_zero_boundary", tea_leaf_zero_boundary_device, -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);
    compileKernel(options, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_init_jac_diag", tea_leaf_init_jac_diag_device, -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);

    if (!rank)
    {
        fprintf(stdout, "done.\n");
        fprintf(DBGOUT, "All kernels compiled\n");
    }
}

CloverChunk::launch_specs_t CloverChunk::findPaddingSize
(int vmin, int vmax, int hmin, int hmax)
{
    int global_horz_size = (-(hmin)) + (hmax) + x_max;
    while (global_horz_size % LOCAL_X) global_horz_size++;
    int global_vert_size = (-(vmin)) + (vmax) + y_max;
    while (global_vert_size % LOCAL_Y) global_vert_size++;
    launch_specs_t cur_specs;
    cur_specs.global = cl::NDRange(global_horz_size, global_vert_size);
    cur_specs.offset = cl::NDRange((halo_exchange_depth) + (hmin), (halo_exchange_depth) + (vmin));
    return cur_specs;
}

void CloverChunk::compileKernel
(std::stringstream& options_orig_knl,
 const std::string& source_name,
 const char* kernel_name,
 cl_kernel& kernel,
 int launch_x_min, int launch_x_max,
 int launch_y_min, int launch_y_max)
{
    std::string source_str;

    kernel_calls[kernel_name] = 0;
    kernel_times[kernel_name] = 0;

    {
        std::ifstream ifile(source_name.c_str());
        source_str = std::string(
            (std::istreambuf_iterator<char>(ifile)),
            (std::istreambuf_iterator<char>()));
    }

    std::stringstream options_orig;
    options_orig << options_orig_knl.str();

    options_orig << "-D KERNEL_X_MIN=" << launch_x_min << " ";
    options_orig << "-D KERNEL_X_MAX=" << launch_x_max << " ";
    options_orig << "-D KERNEL_Y_MIN=" << launch_y_min << " ";
    options_orig << "-D KERNEL_Y_MAX=" << launch_y_max << " ";

    std::string kernel_additional = std::string(kernel_name) + std::string("_device");
    launch_specs[kernel_additional] = findPaddingSize(launch_x_min, launch_x_max, launch_y_min, launch_y_max);

    fprintf(DBGOUT, "Compiling %s...", kernel_name);
    cl_program program;

#if defined(PHI_SOURCE_PROFILING)
    std::stringstream plusprof("");

    if (desired_type == CL_DEVICE_TYPE_ACCELERATOR)
    {
        plusprof << " -profiling ";
        plusprof << " -s \"" << source_name << "\"";
    }
    plusprof << options_orig;
    std::string options(plusprof.str());
#else
    std::string options(options_orig.str());
#endif

    if (built_programs.find(source_name + options) == built_programs.end())
    {
        // Create the program.
        const char *aocx_file_name = "enum_cg";
        std::string binary_file = getBoardBinaryFile(aocx_file_name, device);
        printf("Using AOCX: %s\n\n", binary_file.c_str());
        program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

        // Build the program that was just created.
        status = clBuildProgram(program, 0, NULL, "", NULL, NULL);

        built_programs[source_name + options] = program;
    }
    else
    {
        // + options to stop reduction kernels using the wrong types
        program = built_programs.at(source_name + options);
    }


    try
    {
        printf("Creating kernel[%d]: %s\n", i, kernel_name);
        kernel = clCreateKernel(program, kernel_name, &status);
    }
    catch (cl::Error e)
    {
        fprintf(DBGOUT, "Failed\n");
        DIE("Error %d (%s) in creating %s kernel\n",
            e.err(), e.what(), kernel_name);
    }

    size_t max_wg_size;

    cl::detail::errHandler(
        clGetKernelWorkGroupInfo(kernel,
                                 device,
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),
                                 &max_wg_size,
                                 NULL));
    if ((LOCAL_X*LOCAL_Y) > max_wg_size)
    {
        DIE("Work group size %dx%d is too big for kernel %s"
            " - maximum is %zu\n",
                int(LOCAL_X), int(LOCAL_Y), kernel_name,
                max_wg_size);
    }

    fprintf(DBGOUT, "Done\n");
    fflush(DBGOUT);
}

bool CloverChunk::fileExists(const char *file_name) {
    return access(file_name, R_OK) != -1;
}

// Returns the device name.
std::string CloverChunk::getDeviceName(cl_device_id did) {
    cl_int status;

    size_t sz;
    status = clGetDeviceInfo(did, CL_DEVICE_NAME, 0, NULL, &sz);
    // checkError(status, "Failed to get device name size");

    scoped_array<char> name(sz);
    status = clGetDeviceInfo(did, CL_DEVICE_NAME, sz, name, NULL);
    // checkError(status, "Failed to get device name");

    return name.get();
}

std::string CloverChunk::getBoardBinaryFile(const char *prefix, cl_device_id device) {
    // First check if <prefix>.aocx exists. Use it if it does.
    std::string file_name = std::string(prefix) + ".aocx";
    if(fileExists(file_name.c_str())) {
        return file_name;
    }

    // Now get the name of the board. For Intel(R) FPGA SDK for OpenCL(TM) boards,
    // the name of the device is presented as:
    //  <board name> : ...
    std::string device_name = getDeviceName(device);

    // Now search for the " :" in the device name.
    size_t end = device_name.find(" :");
    if(end != std::string::npos) {
        std::string board_name(device_name, 0, end);

        // Look for a AOCX with the name <prefix>_<board_name>_<version>.aocx.
        file_name = std::string(prefix) + "_" + board_name + "_" + VERSION_STR + ".aocx";
        if(fileExists(file_name.c_str())) {
            return file_name;
        }
    }

    // At this point just use <prefix>.aocx. This file doesn't exist
    // and this should trigger an error later.
    return std::string(prefix) + ".aocx";
}


// Loads a file in binary form.
unsigned char *CloverChunk::loadBinaryFile(const char *file_name, size_t *size) {
    // Open the File
    FILE* fp;
    long ftell_size;
    size_t elements_read;
    fp = fopen(file_name, "rb");
    if(fp == 0) {
        return NULL;
    }

    // Get the size of the file
    fseek(fp, 0, SEEK_END);
    ftell_size = ftell(fp);
    if (ftell_size < 0) {
        fclose(fp);
        return NULL;
    }
    *size = (unsigned)ftell_size;

    // Allocate space for the binary
    unsigned char *binary = new unsigned char[*size];

    // Go back to the file start
    rewind(fp);

    // Read the file into the binary
    elements_read = fread((void*)binary, *size, 1, fp);
    if(elements_read == 0) {
        delete[] binary;
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return binary;
}

// Create a program for all devices associated with the context.
cl_program CloverChunk::createProgramFromBinary(cl_context context, const char *binary_file_name, const cl_device_id *devices, unsigned num_devices) {
    // Early exit for potentially the most common way to fail: AOCX does not exist.
    if(!fileExists(binary_file_name)) {
        printf("AOCX file '%s' does not exist.\n", binary_file_name);
        // checkError(CL_INVALID_PROGRAM, "Failed to load binary file");
    }

    // Load the binary.
    size_t binary_size;
    scoped_array<unsigned char> binary(loadBinaryFile(binary_file_name, &binary_size));
    if(binary == NULL) {
        // checkError(CL_INVALID_PROGRAM, "Failed to load binary file");
    }

    scoped_array<size_t> binary_lengths(num_devices);
    scoped_array<unsigned char *> binaries(num_devices);
    for(unsigned i = 0; i < num_devices; ++i) {
        binary_lengths[i] = binary_size;
        binaries[i] = binary;
    }

    cl_int status;
    scoped_array<cl_int> binary_status(num_devices);

    cl_program program = clCreateProgramWithBinary(context, num_devices, devices, binary_lengths,
            (const unsigned char **) binaries.get(), binary_status, &status);
    // checkError(status, "Failed to create program with binary");
    for(unsigned i = 0; i < num_devices; ++i) {
        // checkError(binary_status[i], "Failed to load binary for device");
    }

    return program;
}

void CloverChunk::initSizes
(void)
{
    fprintf(DBGOUT, "Local size = %dx%d\n", int(LOCAL_X), int(LOCAL_Y));

    // pad the global size so the local size fits
    const int glob_x = x_max+4 +
        (((x_max+4)%LOCAL_X == 0) ? 0 : (LOCAL_X - ((x_max+4)%LOCAL_X)));
    const int glob_y = y_max+4 +
        (((y_max+4)%LOCAL_Y == 0) ? 0 : (LOCAL_Y - ((y_max+4)%LOCAL_Y)));

    fprintf(DBGOUT, "Global size = %dx%d\n", glob_x, glob_y);
    global_size = cl::NDRange(glob_x, glob_y);

    /*
     *  all the reductions only operate on the inner cells, because the halo
     *  cells aren't really part of the simulation. create a new global size
     *  that doesn't include these halo cells for the reduction which should
     *  speed it up a bit
     */
    const int red_x = x_max +
        (((x_max)%LOCAL_X == 0) ? 0 : (LOCAL_X - ((x_max)%LOCAL_X)));
    const int red_y = y_max +
        (((y_max)%LOCAL_Y == 0) ? 0 : (LOCAL_Y - ((y_max)%LOCAL_Y)));
    reduced_cells = red_x*red_y;

    /*
     *  update halo kernels need specific work group sizes - not doing a
     *  reduction, so can just fit it to the row/column even if its not a pwoer
     *  of 2
     */
    // get max local size for the update kernels
    size_t max_update_wg_sz;
    cl::detail::errHandler(
        clGetKernelWorkGroupInfo(update_halo_bottom_device(),
                                 device,
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),
                                 &max_update_wg_sz,
                                 NULL));
    fprintf(DBGOUT, "Max work group size for update halo is %zu\n", max_update_wg_sz);

    // ideally multiple of 32 for nvidia, ideally multiple of 64 for amd
    size_t local_row_size = 64;
    size_t local_column_size = 64;

    cl_device_type dtype;
    status = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);

    if (dtype == CL_DEVICE_TYPE_ACCELERATOR)
    {
        // want to run with work group size of 16 for phi to speed up l/r updates
        local_row_size = 16;
        local_column_size = 16;
    }

    // create the local sizes, dividing the last possible dimension if needs be
    update_lr_local_size[1] = cl::NDRange(1, local_column_size);
    update_lr_local_size[2] = cl::NDRange(2, local_column_size);
    update_bt_local_size[1] = cl::NDRange(local_row_size, 1);
    update_bt_local_size[2] = cl::NDRange(local_row_size, 2);

    // start off doing minimum amount of work
    size_t global_bt_update_size = x_max + 4;
    size_t global_lr_update_size = y_max + 4;

    // increase just to fit in with local work group sizes
    while (global_bt_update_size % local_row_size)
        global_bt_update_size++;
    while (global_lr_update_size % local_column_size)
        global_lr_update_size++;

    // create ndranges for depth 1 and 2
    update_lr_global_size[1] = cl::NDRange(1, global_lr_update_size);
    update_lr_global_size[2] = cl::NDRange(2, global_lr_update_size);
    update_bt_global_size[1] = cl::NDRange(global_bt_update_size, 1);
    update_bt_global_size[2] = cl::NDRange(global_bt_update_size, 2);

    size_t global_bt_pack_size = x_max + 2*halo_exchange_depth;
    size_t global_lr_pack_size = y_max + 2*halo_exchange_depth;

    // increase just to fit in with local work group sizes
    while (global_bt_pack_size % local_row_size)
        global_bt_pack_size++;
    while (global_lr_pack_size % local_column_size)
        global_lr_pack_size++;

    update_lr_global_size[halo_exchange_depth] = cl::NDRange(halo_exchange_depth, global_lr_pack_size);
    update_bt_global_size[halo_exchange_depth] = cl::NDRange(global_bt_pack_size, halo_exchange_depth);

    // use same local size as depth 1
    update_lr_local_size[halo_exchange_depth] = update_lr_local_size[1];
    update_bt_local_size[halo_exchange_depth] = update_bt_local_size[1];

    //for (int depth = 0; depth < 2; depth++)
    std::map<int, cl::NDRange>::iterator typedef irangeit;
    for (irangeit key = update_lr_global_size.begin();
        key != update_lr_global_size.end(); key++)
    {
        int depth = key->first;

        update_lr_offset[depth] = cl::NDRange(halo_exchange_depth - depth, halo_exchange_depth - depth);
        update_bt_offset[depth] = cl::NDRange(halo_exchange_depth - depth, halo_exchange_depth - depth);

        fprintf(DBGOUT, "Depth %d:\n", depth);
        fprintf(DBGOUT, "Left/right update halo size: [%zu %zu] split by [%zu %zu], offset [%zu %zu]\n",
            update_lr_global_size[depth][0], update_lr_global_size[depth][1],
            update_lr_local_size[depth][0], update_lr_local_size[depth][1],
            update_lr_offset[depth][0], update_lr_offset[depth][1]);
        fprintf(DBGOUT, "Bottom/top update halo size: [%zu %zu] split by [%zu %zu], offset [%zu %zu]\n",
            update_bt_global_size[depth][0], update_bt_global_size[depth][1],
            update_bt_local_size[depth][0], update_bt_local_size[depth][1],
            update_bt_offset[depth][0], update_bt_offset[depth][1]);
    }

    fprintf(DBGOUT, "Update halo parameters calculated\n");
}

void CloverChunk::initArgs
(void)
{
    #define SETARG_CHECK(knl, idx, buf) \
        try \
        { \
            knl.setArg(idx, buf); \
        } \
        catch (cl::Error e) \
        { \
            DIE("Error in setting argument index %d to %s for kernel %s (%s - %d)", \
                idx, #buf, #knl, \
                e.what(), e.err()); \
        }

    // initialise chunk
    initialise_chunk_first_device.setArg(4, vertexx);
    initialise_chunk_first_device.setArg(5, vertexdx);
    initialise_chunk_first_device.setArg(6, vertexy);
    initialise_chunk_first_device.setArg(7, vertexdy);
    initialise_chunk_first_device.setArg(8, cellx);
    initialise_chunk_first_device.setArg(9, celldx);
    initialise_chunk_first_device.setArg(10, celly);
    initialise_chunk_first_device.setArg(11, celldy);

    initialise_chunk_second_device.setArg(4, volume);
    initialise_chunk_second_device.setArg(5, xarea);
    initialise_chunk_second_device.setArg(6, yarea);

    // set field
    set_field_device.setArg(0, energy0);
    set_field_device.setArg(1, energy1);

    // generate chunk
    generate_chunk_init_device.setArg(0, density);
    generate_chunk_init_device.setArg(1, energy0);

    generate_chunk_init_u_device.setArg(0, density);
    generate_chunk_init_u_device.setArg(2, u);
    generate_chunk_init_u_device.setArg(3, u0);

    generate_chunk_device.setArg(0, vertexx);
    generate_chunk_device.setArg(1, vertexy);
    generate_chunk_device.setArg(2, cellx);
    generate_chunk_device.setArg(3, celly);
    generate_chunk_device.setArg(4, density);
    generate_chunk_device.setArg(5, energy0);

    // field summary
    field_summary_device.setArg(0, volume);
    field_summary_device.setArg(1, density);
    field_summary_device.setArg(2, energy1);
    field_summary_device.setArg(3, u);

    field_summary_device.setArg(4, reduce_buf_1);
    field_summary_device.setArg(5, reduce_buf_2);
    field_summary_device.setArg(6, reduce_buf_3);
    field_summary_device.setArg(7, reduce_buf_4);

    // no parameters set for update_halo here

    // tealeaf
    if (tea_solver == TEA_ENUM_CG ||
    tea_solver == TEA_ENUM_CHEBYSHEV ||
    tea_solver == TEA_ENUM_PPCG)
    {
        /*
         *  reduce_buf_1 = bb
         *  reduce_buf_2 = rro
         *  reduce_buf_3 = pw
         *  reduce_buf_5 = rrn
         */

        tea_leaf_cg_solve_init_p_device.setArg(0, vector_p);
        tea_leaf_cg_solve_init_p_device.setArg(1, vector_r);
        tea_leaf_cg_solve_init_p_device.setArg(2, vector_z);
        tea_leaf_cg_solve_init_p_device.setArg(3, vector_Mi);
        tea_leaf_cg_solve_init_p_device.setArg(4, reduce_buf_2);

        tea_leaf_cg_solve_calc_w_device.setArg(0, reduce_buf_3);
        tea_leaf_cg_solve_calc_w_device.setArg(1, vector_p);
        tea_leaf_cg_solve_calc_w_device.setArg(2, vector_w);
        tea_leaf_cg_solve_calc_w_device.setArg(3, vector_Kx);
        tea_leaf_cg_solve_calc_w_device.setArg(4, vector_Ky);

        tea_leaf_cg_solve_calc_ur_device.setArg(1, u);
        tea_leaf_cg_solve_calc_ur_device.setArg(2, vector_p);
        tea_leaf_cg_solve_calc_ur_device.setArg(3, vector_r);
        tea_leaf_cg_solve_calc_ur_device.setArg(4, vector_w);
        tea_leaf_cg_solve_calc_ur_device.setArg(5, vector_z);
        tea_leaf_cg_solve_calc_ur_device.setArg(6, cp);
        tea_leaf_cg_solve_calc_ur_device.setArg(7, bfp);
        tea_leaf_cg_solve_calc_ur_device.setArg(8, vector_Mi);
        tea_leaf_cg_solve_calc_ur_device.setArg(9, vector_Kx);
        tea_leaf_cg_solve_calc_ur_device.setArg(10, vector_Ky);
        tea_leaf_cg_solve_calc_ur_device.setArg(11, reduce_buf_5);

        tea_leaf_cg_solve_calc_p_device.setArg(1, vector_p);
        tea_leaf_cg_solve_calc_p_device.setArg(2, vector_r);
        tea_leaf_cg_solve_calc_p_device.setArg(3, vector_z);

        if (tea_solver == TEA_ENUM_CHEBYSHEV)
        {
            tea_leaf_cheby_solve_init_p_device.setArg(0, u);
            tea_leaf_cheby_solve_init_p_device.setArg(1, u0);
            tea_leaf_cheby_solve_init_p_device.setArg(2, vector_p);
            tea_leaf_cheby_solve_init_p_device.setArg(3, vector_r);
            tea_leaf_cheby_solve_init_p_device.setArg(4, vector_w);
            tea_leaf_cheby_solve_init_p_device.setArg(5, cp);
            tea_leaf_cheby_solve_init_p_device.setArg(6, bfp);
            tea_leaf_cheby_solve_init_p_device.setArg(7, vector_Mi);
            tea_leaf_cheby_solve_init_p_device.setArg(8, vector_Kx);
            tea_leaf_cheby_solve_init_p_device.setArg(9, vector_Ky);

            tea_leaf_cheby_solve_calc_u_device.setArg(0, u);
            tea_leaf_cheby_solve_calc_u_device.setArg(1, vector_p);

            tea_leaf_cheby_solve_calc_p_device.setArg(0, u);
            tea_leaf_cheby_solve_calc_p_device.setArg(1, u0);
            tea_leaf_cheby_solve_calc_p_device.setArg(2, vector_p);
            tea_leaf_cheby_solve_calc_p_device.setArg(3, vector_r);
            tea_leaf_cheby_solve_calc_p_device.setArg(4, vector_w);
            tea_leaf_cheby_solve_calc_p_device.setArg(5, cp);
            tea_leaf_cheby_solve_calc_p_device.setArg(6, bfp);
            tea_leaf_cheby_solve_calc_p_device.setArg(7, vector_Mi);
            tea_leaf_cheby_solve_calc_p_device.setArg(8, vector_Kx);
            tea_leaf_cheby_solve_calc_p_device.setArg(9, vector_Ky);
        }
        else if (tea_solver == TEA_ENUM_PPCG)
        {
            tea_leaf_ppcg_solve_init_sd_device.setArg(0, vector_r);
            tea_leaf_ppcg_solve_init_sd_device.setArg(1, vector_sd);
            tea_leaf_ppcg_solve_init_sd_device.setArg(2, vector_z);
            tea_leaf_ppcg_solve_init_sd_device.setArg(3, cp);
            tea_leaf_ppcg_solve_init_sd_device.setArg(4, bfp);
            tea_leaf_ppcg_solve_init_sd_device.setArg(5, vector_Mi);
            tea_leaf_ppcg_solve_init_sd_device.setArg(6, vector_Kx);
            tea_leaf_ppcg_solve_init_sd_device.setArg(7, vector_Ky);
            tea_leaf_ppcg_solve_init_sd_device.setArg(8, u);
            tea_leaf_ppcg_solve_init_sd_device.setArg(9, u0);

            tea_leaf_ppcg_solve_update_r_device.setArg(0, u);
            tea_leaf_ppcg_solve_update_r_device.setArg(1, vector_r);
            tea_leaf_ppcg_solve_update_r_device.setArg(2, vector_Kx);
            tea_leaf_ppcg_solve_update_r_device.setArg(3, vector_Ky);
            tea_leaf_ppcg_solve_update_r_device.setArg(4, vector_sd);

            tea_leaf_ppcg_solve_calc_sd_device.setArg(0, vector_r);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(1, vector_sd);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(2, vector_z);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(3, cp);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(4, bfp);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(5, vector_Mi);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(6, vector_Kx);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(7, vector_Ky);
        }
    }
    else
    {
        // reusing Mi here as 'un'
        tea_leaf_jacobi_copy_u_device.setArg(0, u);
        tea_leaf_jacobi_copy_u_device.setArg(1, vector_Mi);

        tea_leaf_jacobi_solve_device.setArg(0, vector_Kx);
        tea_leaf_jacobi_solve_device.setArg(1, vector_Ky);
        tea_leaf_jacobi_solve_device.setArg(2, u0);
        tea_leaf_jacobi_solve_device.setArg(3, u);
        tea_leaf_jacobi_solve_device.setArg(4, vector_Mi);
        tea_leaf_jacobi_solve_device.setArg(5, reduce_buf_1);
    }

    tea_leaf_calc_residual_device.setArg(0, u);
    tea_leaf_calc_residual_device.setArg(1, u0);
    tea_leaf_calc_residual_device.setArg(2, vector_r);
    tea_leaf_calc_residual_device.setArg(3, vector_Kx);
    tea_leaf_calc_residual_device.setArg(4, vector_Ky);

    tea_leaf_calc_2norm_device.setArg(2, reduce_buf_1);

    // both finalise the same
    tea_leaf_finalise_device.setArg(0, density);
    tea_leaf_finalise_device.setArg(1, u);
    tea_leaf_finalise_device.setArg(2, energy1);

    tea_leaf_init_common_device.setArg(0, density);
    tea_leaf_init_common_device.setArg(1, energy1);
    tea_leaf_init_common_device.setArg(2, vector_Kx);
    tea_leaf_init_common_device.setArg(3, vector_Ky);
    tea_leaf_init_common_device.setArg(4, u0);
    tea_leaf_init_common_device.setArg(5, u);

    // block
    tea_leaf_block_init_device.setArg(0, vector_r);
    tea_leaf_block_init_device.setArg(1, vector_z);
    tea_leaf_block_init_device.setArg(2, cp);
    tea_leaf_block_init_device.setArg(3, bfp);
    tea_leaf_block_init_device.setArg(4, vector_Kx);
    tea_leaf_block_init_device.setArg(5, vector_Ky);

    tea_leaf_block_solve_device.setArg(0, vector_r);
    tea_leaf_block_solve_device.setArg(1, vector_z);
    tea_leaf_block_solve_device.setArg(2, cp);
    tea_leaf_block_solve_device.setArg(3, bfp);
    tea_leaf_block_solve_device.setArg(4, vector_Kx);
    tea_leaf_block_solve_device.setArg(5, vector_Ky);

    tea_leaf_init_jac_diag_device.setArg(0, vector_Mi);
    tea_leaf_init_jac_diag_device.setArg(1, vector_Kx);
    tea_leaf_init_jac_diag_device.setArg(2, vector_Ky);

    fprintf(DBGOUT, "Kernel arguments set\n");
}

