#include "definitions.hpp"

#define x_max 10
#define y_max 10
#define HALO_DEPTH 4
#define BLOCK_SZ LOCAL_X*LOCAL_Y

#define CL_DEVICE_TYPE_ACCELERATOR

// #define red_sum
// typedef double reduce_t;
// #define INIT_RED_VAL 0

#define PRECONDITIONER TEA_ENUM_CG

#include "initialise_chunk_cl.cl"
#include "generate_chunk_cl.cl"
#include "set_field_cl.cl"
#include "field_summary_cl.cl"
#include "update_halo_cl.cl"
#include "pack_kernel_cl.cl"
#include "tea_leaf_common_cl.cl"
#include "tea_leaf_cg_cl.cl"
