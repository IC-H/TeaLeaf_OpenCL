#ifndef __CL_COMMON_HDR
#define __CL_COMMON_HDR

#include "CL/cl.hpp"

#include <cstdio>
#include <cstdlib>

#include "mpi.h"

#include "kernel_files/definitions.hpp"
#include "types.hpp"

extern TeaCLContext tea_context;

// this function gets called when something goes wrong
#define DIE(...) cloverDie(__LINE__, __FILE__, __VA_ARGS__)
extern void cloverDie
(int line, const char* filename, const char* format, ...);

#endif
