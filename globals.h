#ifndef __GLOBALS_H
#define __GLOBALS_H

// Implementation by kernelPBO.cu (compiled with nvcc)
extern "C" void launch_kernel(uchar4*, unsigned int, unsigned int, int);

// --------------------------------------------------------------------------
/**
 * Error handling for cuda functions.
 * Use macro HANDLE_ERROR( <cuda_function> ).
 * @param err cudaError_t CUDA Error type
 * @param file source file (macro calls by __FILE__)
 * @param line source line (macro calls by __LINE__)
 * 
 * Reference: http://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming
 */
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s (%d) in %s at line %d\n", cudaGetErrorString( err ), err, 
                file, line );
        exit( EXIT_FAILURE );
    }
}

// macro for error handling cuda functions
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif
