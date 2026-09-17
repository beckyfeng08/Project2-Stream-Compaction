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
        int blockSize = 64;


        __global__ void upsweep(int n, int offset, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int two_dplusone = 1 << (offset + 1);
            int k = index * two_dplusone;
             if (k + two_dplusone - 1 >= n) {
                return;
            }
            int two_d = 1 << offset;
            idata[k + two_dplusone - 1] += idata[k + two_d - 1]; 
            
        }

        __global__ void downsweep(int n, int offset, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int two_dplusone = 1 << (offset + 1);
            int k = index * two_dplusone;
            if  (k + two_dplusone - 1 >= n) {
                return;
            }
            int two_d = 1 << offset;

            int t = idata[k + two_d - 1];
            idata[k + two_d - 1] = idata[k + two_dplusone - 1];
            idata[k + two_dplusone - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // setting block size
            int fullBlocksPerGrid((n +  blockSize - 1) /  blockSize);

            int* idata2;
            int n_padded = n;

            // check to see if we have a power of 2. If not, resize and pad with 0s
            if (1 << ilog2ceil(n) != n) {
                n_padded = 1 << ilog2ceil(n);
            }

            cudaMalloc((void**)&idata2, n_padded * sizeof(int));
            // copy values and pad with 0s
            cudaMemset(idata2, 0, n_padded * sizeof(int));
            cudaMemcpy(idata2, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                        timer().startGpuTimer();

            int logn = ilog2ceil(n);

            // upsweep
            for (int d = 0; d < logn; d++) {
                int numelements = 1 << (logn - d);
                fullBlocksPerGrid =( numelements +  blockSize - 1) /  blockSize;
                upsweep<<<fullBlocksPerGrid,  blockSize>>>(n_padded, d, idata2);
            }
            // downsweep, geenrates exclusive scan
            cudaMemset(idata2 + n_padded - 1, 0, sizeof(int));
            for (int d = logn - 1; d >= 0; d--) {
                int numelements = 1 << (logn - d);
                fullBlocksPerGrid =( numelements +  blockSize - 1) /  blockSize;
                downsweep<<<fullBlocksPerGrid,  blockSize>>>(n_padded, d, idata2);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, idata2, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(idata2);
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



        int compact(int n, int *odata, const int *idata) {
            //timer().startGpuTimer();
            int fullBlocksPerGrid((n +  blockSize - 1) /  blockSize);

            // TODO
            int* booldata;
            int* booldata_scanned;

            int* idata2;
            int* odata2;

            cudaMalloc(&booldata, n * sizeof(int));
            cudaMemset(booldata, 0, n * sizeof(int));

            cudaMalloc((void**) &booldata_scanned, n * sizeof(int));
            cudaMemset(booldata_scanned, 0, n * sizeof(int));

            cudaMalloc(&idata2, n * sizeof(int));
            cudaMalloc(&odata2, n * sizeof(int));
            cudaMemcpy(idata2, idata, n * sizeof(int), cudaMemcpyHostToDevice); // must be put on device i think for this to work

            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid,  blockSize >> > (n, booldata, idata2);

            scan(n, booldata_scanned, booldata);

            StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid,  blockSize>>>(n, odata2, idata2, booldata, booldata_scanned);

            int count1;
            int count2;

            cudaMemcpy( &count1, booldata_scanned + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&count2, booldata + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(odata, odata2, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(booldata);
            cudaFree(booldata_scanned);
            cudaFree(odata2);
            cudaFree(idata2);

            //timer().endGpuTimer();

            return count1 + count2;
        }
    }
}
