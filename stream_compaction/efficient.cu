#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void padzeros(int startidx, int endidx, int* a) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < endidx - startidx) {
                a[startidx + index] = 0;
            }
        }
        __global__ void upsweep(int n, int offset, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                int two_dplusone = 1 << (offset + 1);
                int two_d = 1 << offset;

                idata[index + two_dplusone - 1] += idata[index + two_d - 1];
            }
        }

        __global__ void downsweep(int n, int offset, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                int two_d = 1 << offset;
                int two_dplusone = 1 << (offset + 1);

                int t = idata[index + two_d - 1];
                idata[index + two_d - 1] = idata[index + two_dplusone - 1];
                idata[index + two_dplusone - 1] += t;
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

            int* idata2;
            int n_padded = n;

            // check to see if we have a power of 2. If not, resize and pad with 0s
            if (1 << ilog2ceil(n) != n) {
                n_padded = 1 << ilog2ceil(n);
            }

            cudaMalloc((void**)&idata2, n_padded * sizeof(int));
            cudaMemcpy(idata2, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            padzeros<<<fullBlocksPerGrid, blockSize>>>(n, n_padded, idata2);

            // upsweep
            for (int d = 0; d < ilog2ceil(n) - 1 ; d++) {
                upsweep<<<fullBlocksPerGrid, blockSize>>>(n_padded, d, idata2);
            }
            // downsweep, geenrates exclusive scan
            cudaMemset(idata2 + n_padded - 1, 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d > 0; d--) {
                downsweep<<<fullBlocksPerGrid, blockSize>>>(n_padded, d, idata2);
            }

            cudaMemcpy(odata, idata2, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(idata2);
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */

         __global__ void genBoolArray(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (idata[index] == 0) return;
                odata[index] = 1;
            }
         }

          __global__ void scatter(int n, int* booldata_scanned, int* idata, int* odata) {
            int count = 0;
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n - 1) {
               // we write if the right is a new number
               if (booldata_scanned[index + 1] != booldata_scanned[index]) {
                odata[index] = idata[index];
                count++;
               }
            }
         }

        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int blockSize = 64;
            int fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // TODO
            int* booldata;
            int* booldata_scanned;
            cudaMalloc((void**) &booldata, n * sizeof(int));
            cudaMalloc((void**) &booldata_scanned, n * sizeof(int));

            genBoolArray<<<fullBlocksPerGrid, blockSize>>>(n, booldata, idata);
            scan(n, booldata_scanned, booldata);

            scatter<<<fullBlocksPerGrid, blockSize>>>(n, booldata_scanned, idata, odata);
            int count = booldata_scanned[n - 1];

            cudaFree(booldata);
            cudaFree(booldata_scanned);
        

            timer().endGpuTimer();

            return count;
        }
    }
}
