#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = count;
                count += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int j = 0; // index for odata
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) { // find nonzero elements
                    odata[j] = idata[i]; // add into odata
                    j++; // increment
                }
            }
            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            //timer().startCpuTimer();
            // TODO
            // map original data array
            int *prescan = new int[n]; //  populate with zeros
            int *scanresult = new int[n];
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0)
                    prescan[i] = 0;
                else
                    prescan[i] = 1;
            }
            // scan
            scan(n, scanresult, prescan);
            //scatter
            int curridx = 0;
            for (int i = 0; i < n; i++) {
                if (prescan[i] == 1) {
                    int idx = scanresult[i];
                    odata[idx] = idata[i];
                    curridx++;
                }
            }
       
            delete[] prescan;
            delete[] scanresult;
            //timer().endCpuTimer();
            return curridx;
        }
    }
}
