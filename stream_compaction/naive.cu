#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scanHelper(int n, int offset, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (index >= offset)
                    odata[index] = idata[index - offset] + idata[index];
                else
                    odata[index] = idata[index];
            }
        }

        __global__ void inclusive2exclusive(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (index == 0) 
                    odata[index] = 0;
                else
                    odata[index] = idata[index - 1];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            // setting block size
            int blockSize = 64;
            int fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // initialize helper buffers for swapping
            int* idata2;
            int* odata2;

            cudaMalloc((void**)&idata2, n * sizeof(int));
            cudaMalloc((void**)&odata2, n * sizeof(int));

            // copy contents
            cudaMemcpy(idata2, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int lvlcount = ilog2ceil(n);

            for (int d = 0; d < lvlcount; d++) {
                int offset = 1 << d; // 2^d-1

                scanHelper<<<fullBlocksPerGrid, blockSize>> > (n, offset, odata2, idata2);
                // swap buffers
                int* tmp = idata2;
                idata2 = odata2;
                odata2 = tmp;
            }
            // inclusive to exclusive
            inclusive2exclusive<< <fullBlocksPerGrid, blockSize >> > (n, odata2, idata2);

            // at the end, populate odata
            cudaMemcpy(odata, odata2, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(idata2);
            cudaFree(odata2);
            
            timer().endGpuTimer();
        }
    }
}
