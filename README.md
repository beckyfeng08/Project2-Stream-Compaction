CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Rebecca Feng
  * = [LinkedIn](https://www.linkedin.com/in/beckyfeng0803/), [personal website](https://beckyfeng08.github.io/)
Tested on: Windows 11, AMD Ryzen 9 @ 2.50 GHz 8GB, GTX 5060
Visual Studio 2022, CUDA 13.3

## Description

This repo explores implementations of stream compaction and scan/prefix sums. 

Scan implementations include:
- CPU: Uses a for loop to compute the sum in a given array
- Naive: On the GPU. Builds partial sums by adding some index i and i - 2^(d - 1)th together, where d is dth pass of adding, with log_2(n) total passes until we sum up all the elements of the array.
- Work-efficient: On the GPU. Builds partial sums by modifying the array in-place via upsweeping and downsweeping.
- Thrust: Using the Thrust library's version of scan to compare performance between our implementation vs theirs

Stream compaction implementations include:
- CPU without scan: Stream compaction without the scan function where we loop over all elements of the array and populate the output array whenever data[i] != 0
- CPU with scan: Invokes the scan function - sets up an array of 0's and 1's depending on if the elements in the input array are "valid", applies the scan function to it to receive the indices necessary for copying to the output array
- Work-efficient: Same as previous, but on the GPU and invokes work-efficient scan function.

### Evaluating performance
All of the following data were taken on Release mode, for array sizes of power-of-twos.

#### CPU and GPU Scan Performance Results
| Size of Array (log_2 scale) | CPU (ms) | Naive (ms) | Work-Efficient (ms) | Thrust (ms) |
| ------------: | -------: | ---------: | ------------------: | ----------: |
|             9 |   0.0009 |      0.379 |               0.536 |       3.312 |
|            10 |   0.0012 |      0.519 |               0.626 |       3.094 |
|            15 |   0.0204 |      1.409 |               2.506 |       2.560 |
|            19 |     0.33 |      0.696 |               1.053 |       3.644 |
|            20 |    0.621 |      0.950 |               1.004 |       3.480 |
|            21 |    1.296 |      1.282 |               0.962 |       3.251 |
|            22 |    2.572 |      2.206 |               1.447 |       3.768 |
|            23 |    5.493 |      5.567 |               1.956 |       4.318 |
|            24 |    10.63 |     11.230 |               5.320 |       4.360 |

![scan performance graph](img/Unknown.png)

We see that for large array sizes (greater than 2^21 elements, our work-efficient GPU scan starts to surpass the performance of our CPU scan, which increases linearly). Our naive GPU scan implementation roughly performs similarly to the CPU implementation. Meanwhile, our thrust wrapper performs relatively uniformly wrt time, regardless of how large of an array we feed into it, and starts to perform better than all of our other algorithms after extremely large array sizes, at a size of 2^24. However, it is the most inefficient with smaller sized arrays. The reason why this is the case is due to blah blah blah TODO.

What is causing performance bottlenecks in each computation system?


Release version test output


