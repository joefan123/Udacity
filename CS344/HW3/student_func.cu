/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <cmath>

__global__ void dummy(const float * const d_in)
{
    // printf("d_logLuminance[1]=%f\n", d_in[1]);
    // printf("d_logLuminance[100]=%f\n", d_in[100]);
    // printf("d_logLuminance[10837]=%f\n", d_in[10837]);
    // printf("d_logLuminance[14168]=%f\n", d_in[14168]);

    for (int vCount = 0; vCount < 256; vCount++)
    {
        printf("d_logLuminance[%d]=%f\n", vCount, d_in[vCount]);
    }

}


__global__ void dummy2(const float * const d_in)
{
    for (int vCount = 0; vCount < 5; vCount++)
    {
        printf("d_in[%d]=%f\n", vCount, d_in[vCount]);
    }

}


__global__ void global_CalcMinMax_kernel(const float * const d_logLuminance,
                                         const int pBlockSize,
                                         float * pMin,
                                         float * pMax)
{

    // Declarations
    float vMin, vMax;
    int vPixelID;
    int vCount1;

    // printf("blockIdx.x = %d\n", blockIdx.x);
    // printf("blockDim.x = %d\n", blockDim.x);
    // Initialize
    vMin = +99.0;
    vMax = -99.0;
    vPixelID = blockIdx.x * pBlockSize;

    for (vCount1 = 0; vCount1 < pBlockSize ; vCount1++)
    {
        vMin = fmin(d_logLuminance[vPixelID + vCount1], vMin);
        vMax = fmax(d_logLuminance[vPixelID + vCount1], vMax);
    }
    pMin[blockIdx.x] = vMin;
    pMax[blockIdx.x] = vMax;
    

}

__global__ void global_CalcMinMax2_kernel(float * const d_intermediate_min,
                                          float * const d_intermediate_max,
                                          const int pBlocks)
{
    // Declarations
    float vMin, vMax;
    int vCount1;

    vMin = +99.0;
    vMax = -99.0;
    for (vCount1 = 0; vCount1 < pBlocks ; vCount1++)
    {
        vMin = fmin(d_intermediate_min[vCount1], vMin);
        vMax = fmax(d_intermediate_max[vCount1], vMax);
    }
    d_intermediate_min[0] = vMin;
    d_intermediate_max[0] = vMax;
}

void CalcMinMax(const float * const d_logLuminance,
                int const pNumRows,
                int const pNumCols,
                float &pMin_logLum,
                float &pMax_logLum)
{

    // Assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock

    // Declarations
    bool vUsesSharedMemory;
    int vMaxThreadsPerBlock;
    int vThreads;
    int vBlocks;
    float *d_intermediate_min;
    float *d_intermediate_max;
    int vCount1;

    // Allocate memory
    checkCudaErrors(cudaMalloc(&d_intermediate_min, sizeof(float) * pNumRows * pNumCols));
    checkCudaErrors(cudaMalloc(&d_intermediate_max, sizeof(float) * pNumRows * pNumCols));

    // Initalization
    int vBlockSize = 256;
    vBlocks = pNumRows * pNumCols / vBlockSize;

    // Calc min and max for sections in the picture
    global_CalcMinMax_kernel<<<vBlocks, 1>>>
        (d_logLuminance,
         vBlockSize,
         d_intermediate_min,
         d_intermediate_max);
    
    // debug 1
    dummy2<<<1, 1>>>(d_intermediate_min);
    dummy2<<<1, 1>>>(d_intermediate_max);
    
    // Consolidate min and max values
    global_CalcMinMax2_kernel<<<1,1>>>
        (d_intermediate_min,
         d_intermediate_max,
         vBlocks);
    checkCudaErrors(cudaMemcpy(&pMin_logLum, d_intermediate_min, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&pMax_logLum, d_intermediate_max, sizeof(float), cudaMemcpyDeviceToHost));

    // debug 2
    printf ("pMin_logLum=%f\n", pMin_logLum);
    printf ("pMax_logLum=%f\n", pMax_logLum);
    
    // Free GPU memory allocation
    checkCudaErrors(cudaFree(d_intermediate_min));
    checkCudaErrors(cudaFree(d_intermediate_max));

}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &pMin_logLum,
                                  float &pMax_logLum,
                                  const size_t pNumRows,
                                  const size_t pNumCols,
                                  const size_t pNumBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    printf("pNumRows=%d\n", pNumRows);
    printf("pNumCols=%d\n", pNumCols);
    printf("pNumBins=%d\n", pNumBins);

    // display sample luminance values
    // dummy<<<1, 1>>>(d_logLuminance);
   
    
    // Calc the min and max value of d_logLuminance
    CalcMinMax(d_logLuminance,
               pNumRows,
               pNumCols,
               pMin_logLum,
               pMax_logLum);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
 
}