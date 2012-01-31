#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.1415926535897932f

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
} 

__global__ void kernel( uchar4 *ptr, unsigned int width,
                        unsigned int height, int iter, int time ) {
    // map from threadIdx/BlockIdx to pixel position
    float x = threadIdx.x + blockIdx.x * blockDim.x;
    float y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    float cx = 3.0f * (x / width-0.5f);
    float cy = 2.0f * (y / height-0.5f);

    int i;
    float zx = cx;
    float zy = cy;
    float dx = 0.0f;
    float dy = 0.0f;
    cx *=  cos(time/1000.0f);
    cy *=  sin(time/1000.0f);
    for(i=0; i<iter; i++) {
        dx = (zx * zx - zy * zy) + cx + dx;
        dy = (zy * zx + zx * zy) + cy + dy;
        if((dx * dx + dy * dy) > 4.0f) break;
        zx = dx;
        zy = dy;
    }

    unsigned char val = 255.0f*float(i==iter?0:i)/iter;
    
    ptr[offset].x = min(255,3*val/2);
    ptr[offset].y = (val>100 ? min(255,3*(val-100)) : 0);
    ptr[offset].z = (val>127 ? min(255,2*(val-127)) : 0);
    ptr[offset].w = 255;
}

extern "C" void launch_kernel( uchar4* pos, unsigned int image_width,
                               unsigned int image_height, int time ) {
    dim3    blocks(image_width/16, image_height/16);
    dim3    threads(16,16);

    kernel<<<blocks, threads>>>(pos, image_width, image_height, 48, time);
}


//Simple kernel writes changing colors to a uchar4 array
/*__global__ void kernel(uchar4* pos, unsigned int width, unsigned int height,
               int time)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x = index%width;
  unsigned int y = index/width;
  
  if(index < width*height) {
    unsigned char r = (x + time)&0xff;
    unsigned char g = (y + time)&0xff;
    unsigned char b = ((x+y) + time)&0xff;
    
    // Each thread writes one pixel location in the texture (texel)
    pos[index].w = 0;
    pos[index].x = r;
    pos[index].y = g;
    pos[index].z = b;
  }
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(uchar4* pos, unsigned int image_width, 
                  unsigned int image_height, int time)
{
  // execute the kernel
  int nThreads=256;
  int totalThreads = image_height * image_width;
  int nBlocks = totalThreads/nThreads;
  
  nBlocks += ((totalThreads%nThreads)>0)?1:0;

  kernel<<<nBlocks, nThreads>>>(pos, image_width, image_height, time/10);
  
  // make certain the kernel has completed
  cudaThreadSynchronize();
  
  checkCUDAError("kernel failed!");
}*/
