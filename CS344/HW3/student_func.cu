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


// --------------------------------------------------------------------------------
// Debug output
// --------------------------------------------------------------------------------


__global__ 
void dummy1(const float * const d_in)
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


__global__ 
void dummy2(const float * const d_in)
{
    for (int vCount = 0; vCount < 5; vCount++)
    {
        printf("d_in[%d]=%f\n", vCount, d_in[vCount]);
    }

}

__global__ 
void dummy3(const unsigned int * const pHistogram)
{
    for (int vCount = 0; vCount < 1024; vCount++)
    {
        printf("pHistogram[%d]=%d\n", vCount, pHistogram[vCount]);
    }

}


// --------------------------------------------------------------------------------
// GPU kernels
// --------------------------------------------------------------------------------


__global__ 
void global_CalcMinMax1_kernel(const float * const d_pArrLogLuminance,
                               const int pBlockSize,
                               float * pMin,
                               float * pMax)
{

    // Declarations
    float vMin;
    float vMax;
    int vPixelID;
    int vCount1;

    // Debug
    // printf("blockIdx.x = %d\n", blockIdx.x);
    // printf("blockDim.x = %d\n", blockDim.x);

    // Initialize
    vMin = +99.0;
    vMax = -99.0;
    vPixelID = blockIdx.x * pBlockSize;

    for (vCount1 = 0; vCount1 < pBlockSize ; vCount1++)
    {
        vMin = fmin(d_pArrLogLuminance[vPixelID + vCount1], vMin);
        vMax = fmax(d_pArrLogLuminance[vPixelID + vCount1], vMax);
    }
    pMin[blockIdx.x] = vMin;
    pMax[blockIdx.x] = vMax;
}

__global__ 
void global_CalcMinMax2_kernel(float * const d_pArrIntermediate_min,
                               float * const d_pArrIntermediate_max,
                               const int pBlocks)
{
    // Declarations
    float vMin;
    float vMax;
    int vCount1;

    vMin = +99.0;
    vMax = -99.0;
    for (vCount1 = 0; vCount1 < pBlocks ; vCount1++)
    {
        vMin = fmin(d_pArrIntermediate_min[vCount1], vMin);
        vMax = fmax(d_pArrIntermediate_max[vCount1], vMax);
    }
    d_pArrIntermediate_min[0] = vMin;
    d_pArrIntermediate_max[0] = vMax;
}

__global__
void global_InitHist_kernel(unsigned int * const pArrHistogram)
{
    // Initialize histogram
    pArrHistogram[threadIdx.x] = 0;
}

__global__ 
void global_CalcHistOld_kernel(const float * const d_pArrLogLuminance,
                                  int const pNumRows,
                                  int const pNumCols,
                                  int const pNumBins,
                                  float const pMin_logLum,
                                  float const pMax_logLum,
                                  unsigned int * pArrHistogram)
{

    // Declarations
    int vBinID;
    float vRange;
    int vIdxRow;
    int vIdxCol;
    int vIdxPixel;

    // Initialize pixel location
    vIdxCol = (blockIdx.x * blockDim.x) + threadIdx.x;
    vIdxRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    vIdxPixel = vIdxRow * pNumCols + vIdxCol;

    // Check image bounds before accessing GPU memory
    if ( vIdxCol >= pNumCols || vIdxRow >= pNumRows )
        return;

    // Identify bins
    vRange = pMax_logLum - pMin_logLum;
    vBinID = min(static_cast<unsigned int>(pNumBins - 1),
                 static_cast<unsigned int>((d_pArrLogLuminance[vIdxPixel] - pMin_logLum) / vRange * pNumBins));

    // Debug output
    // printf("vIdxCol = %d, vIdxRow = %d, vIdxPixel = %d, vBinID = %d\n", vIdxCol, vIdxRow, vIdxPixel, vBinID);

    // Increment histogram
    atomicAdd(&(pArrHistogram[vBinID]), 1);

}


__global__ 
void global_CalcHistReduce_kernel(const float * const d_pArrLogLuminance,
                                  int const pNumRows,
                                  int const pNumCols,
                                  int const pNumBins,
                                  float const pMin_logLum,
                                  float const pMax_logLum,
                                  unsigned int * pArrHistogram)
{
    // Declarations
    int vBinID;
    float vRange;
    int vIdxPixel;

    // Allocate memory for thread block's histogram
    extern __shared__ unsigned int sh_vArrHistogram[];

    // Initialize thread block's histogram
    vBinID = threadIdx.x;  
    sh_vArrHistogram[vBinID] = 0;

    __syncthreads();    // ensure that initialization is complete

    // Initialize pixel location
    vIdxPixel = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (vIdxPixel > pNumRows * pNumCols)
        return;

    // Identify bins
    vRange = pMax_logLum - pMin_logLum;
    vBinID = min(static_cast<unsigned int>(pNumBins - 1),
                 static_cast<unsigned int>((d_pArrLogLuminance[vIdxPixel] - pMin_logLum) / vRange * pNumBins));

    // Increment thread block's histogram
    atomicAdd(&sh_vArrHistogram[vBinID], 1);

    __syncthreads();    // ensure that all thread block histograms are complete

    // Consolidate all thread block histograms into final histogram
    vBinID = threadIdx.x;
    atomicAdd(&pArrHistogram[vBinID], sh_vArrHistogram[vBinID]);

}


__global__ 
void global_CalcHistNew_kernel(const float * const d_pArrLogLuminance,
                                  int const pNumRows,
                                  int const pNumCols,
                                  int const pNumBins,
                                  float const pMin_logLum,
                                  float const pMax_logLum,
                                  unsigned int * pArrHistogram)
{

    // Declarations
    int vBinID;
    float vRange;
    int vIdxRow;
    int vIdxCol;
    int vIdxPixel;

    // Initialize pixel location
    vIdxCol = (blockIdx.x * blockDim.x) + threadIdx.x;
    vIdxRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    vIdxPixel = vIdxRow * pNumCols + vIdxCol;

    // Check image bounds before accessing GPU memory
    if ( vIdxCol >= pNumCols || vIdxRow >= pNumRows )
        return;

    // Identify bins
    vRange = pMax_logLum - pMin_logLum;
    vBinID = min(static_cast<unsigned int>(pNumBins - 1),
                 static_cast<unsigned int>((d_pArrLogLuminance[vIdxPixel] - pMin_logLum) / vRange * pNumBins));

    // Debug output
    // printf("vIdxCol = %d, vIdxRow = %d, vIdxPixel = %d, vBinID = %d\n", vIdxCol, vIdxRow, vIdxPixel, vBinID);

    // Increment histogram
    atomicAdd(&(pArrHistogram[vBinID]), 1);

}





// --------------------------------------------------------------------------------
// Subroutine functions
// --------------------------------------------------------------------------------

void CalcMinMax(const float * const d_pArrLogLuminance,
                int const pNumRows,
                int const pNumCols,
                float &pMin_logLum,
                float &pMax_logLum)
{

    // Assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock

    // Declarations
    int vBlocks;
    int vBlockSize;
    float *d_vArrIntermediate_min;
    float *d_vArrIntermediate_max;

    // Allocate memory
    checkCudaErrors(cudaMalloc(&d_vArrIntermediate_min, sizeof(float) * pNumRows * pNumCols));
    checkCudaErrors(cudaMalloc(&d_vArrIntermediate_max, sizeof(float) * pNumRows * pNumCols));

    // Initalization
    vBlockSize = 16 * 16;
    vBlocks = pNumRows * pNumCols / vBlockSize;

    // Calc min and max for sections in the picture
    global_CalcMinMax1_kernel<<<vBlocks, 1>>>
        (d_pArrLogLuminance,
         vBlockSize,
         d_vArrIntermediate_min,
         d_vArrIntermediate_max);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
    // debug 1
    dummy2<<<1, 1>>>(d_vArrIntermediate_min);
    dummy2<<<1, 1>>>(d_vArrIntermediate_max);
    
    // Consolidate min and max values
    global_CalcMinMax2_kernel<<<1,1>>>
        (d_vArrIntermediate_min,
         d_vArrIntermediate_max,
         vBlocks);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(&pMin_logLum, d_vArrIntermediate_min, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&pMax_logLum, d_vArrIntermediate_max, sizeof(float), cudaMemcpyDeviceToHost));

    // debug 2
    printf ("pMin_logLum=%f\n", pMin_logLum);
    printf ("pMax_logLum=%f\n", pMax_logLum);
    
    // Free GPU memory allocation
    checkCudaErrors(cudaFree(d_vArrIntermediate_min));
    checkCudaErrors(cudaFree(d_vArrIntermediate_max));

}

void CalcHistogramOld(const float * const d_pArrLogLuminance,
                   int const pNumRows,
                   int const pNumCols,
                   int const pNumBins,
                   float const pMin_logLum,
                   float const pMax_logLum,
                   unsigned int * pArrHistogram)
{
    // Initalization
    // const dim3 vBlockSize(16, 16, 1);
    const dim3 vBlockSize(32, 32, 1);

    const dim3 vGridSize(ceil(static_cast<float>(pNumCols) / vBlockSize.x), 
                         ceil(static_cast<float>(pNumRows) / vBlockSize.y),
                         1);

    // Initialize histogram
    global_InitHist_kernel<<<1, pNumBins>>>
        (pArrHistogram);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
    // Generate histogram
    global_CalcHistOld_kernel<<<vGridSize, vBlockSize>>>
        (d_pArrLogLuminance,
         pNumRows,
         pNumCols,
         pNumBins,
         pMin_logLum,
         pMax_logLum,
         pArrHistogram);
    
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
  
    // debug
    dummy3<<<1, 1>>>(pArrHistogram);
    checkCudaErrors(cudaGetLastError());

}

void CalcHistogramReduce(const float * const d_pArrLogLuminance,
                         int const pNumRows,
                         int const pNumCols,
                         int const pNumBins,
                         float const pMin_logLum,
                         float const pMax_logLum,
                         unsigned int * pArrHistogram)
{

    // Initalization
    const int vBlockSize = pNumBins;
    const int vGridSize = ceil(static_cast<float>(pNumRows * pNumCols) / pNumBins);

    // Initialize histogram
    global_InitHist_kernel<<<1, pNumBins>>>
        (pArrHistogram);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
    // Generate histogram
    global_CalcHistReduce_kernel<<<vGridSize, vBlockSize, pNumBins * sizeof(unsigned int)>>>
        (d_pArrLogLuminance,
         pNumRows,
         pNumCols,
         pNumBins,
         pMin_logLum,
         pMax_logLum,
         pArrHistogram);
    
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // debug
    dummy3<<<1, 1>>>(pArrHistogram);
    checkCudaErrors(cudaGetLastError());

}


void CalcHistogramNew(const float * const d_pArrLogLuminance,
                         int const pNumRows,
                         int const pNumCols,
                         int const pNumBins,
                         float const pMin_logLum,
                         float const pMax_logLum,
                         unsigned int * pArrHistogram)
{
    printf("Work in progress\n");
    /*
    // Initalization
    // const dim3 vBlockSize(16, 16, 1);
    const dim3 vBlockSize(32, 32, 1);

    const dim3 vGridSize(ceil(static_cast<float>(pNumCols) / vBlockSize.x), 
                         ceil(static_cast<float>(pNumRows) / vBlockSize.y),
                         1);

    // Initialize histogram
    global_InitHist_kernel<<<1, pNumBins>>>
        (pArrHistogram);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
    // Generate histogram
    global_CalcHistNew_kernel<<<vGridSize, vBlockSize>>>
        (d_pArrLogLuminance,
         pNumRows,
         pNumCols,
         pNumBins,
         pMin_logLum,
         pMax_logLum,
         pArrHistogram);
    
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
  
    // debug
    dummy3<<<1, 1>>>(pArrHistogram);
    checkCudaErrors(cudaGetLastError());
    */
}


void CalcCDF(const unsigned int * const pArrHistogram,
             const int pNumBins,
             unsigned int * const d_pArrCDF)
{

    

}

void ShowDeviceProperty()
{

    // Declarations
    cudaDeviceProp deviceProperty;
    int totalCudaCores;

    // Initialize cuda properties
    checkCudaErrors( cudaGetDeviceProperties( &deviceProperty, 0 ) );

    // Show CUDA properties
    printf("device name                = %s\n" , deviceProperty.name);
    printf("device sharedMemPerBlock   = %d\n" , deviceProperty.sharedMemPerBlock);
    printf("device totalGlobalMem      = %d\n" , deviceProperty.totalGlobalMem);
    printf("device regsPerBlock        = %d\n" , deviceProperty.regsPerBlock);
    printf("device warpSize            = %d\n" , deviceProperty.warpSize);
    printf("device memPitch            = %d\n" , deviceProperty.memPitch);
    printf("device maxThreadsPerBlock  = %d\n" , deviceProperty.maxThreadsPerBlock);
    printf("device maxThreadsDim[0]    = %d\n" , deviceProperty.maxThreadsDim[0]);
    printf("device maxThreadsDim[1]    = %d\n" , deviceProperty.maxThreadsDim[1]);
    printf("device maxThreadsDim[2]    = %d\n" , deviceProperty.maxThreadsDim[2]);
    printf("device maxGridSize[0]      = %d\n" , deviceProperty.maxGridSize[0]);
    printf("device maxGridSize[1]      = %d\n" , deviceProperty.maxGridSize[1]);
    printf("device maxGridSize[2]      = %d\n" , deviceProperty.maxGridSize[2]);
    printf("device totalConstMem       = %d\n" , deviceProperty.totalConstMem);
    printf("device major               = %d\n" , deviceProperty.major);
    printf("device minor               = %d\n" , deviceProperty.minor);
    printf("device clockRate           = %d\n" , deviceProperty.clockRate);
    printf("device textureAlignment    = %d\n" , deviceProperty.textureAlignment);
    printf("device deviceOverlap       = %d\n" , deviceProperty.deviceOverlap);
    printf("device multiProcessorCount = %d\n" , deviceProperty.multiProcessorCount);
    
    // Show total cores
    if (deviceProperty.major == 1)
    {
        totalCudaCores = (deviceProperty.multiProcessorCount*8);
        printf("Total CUDA cores: %d \n", totalCudaCores);
    }
    else if (deviceProperty.major == 2)
    {
        if (deviceProperty.minor == 0)
        {
            totalCudaCores = (deviceProperty.multiProcessorCount*32);
            printf("Total CUDA cores: %d \n", totalCudaCores);
        }
        else if (deviceProperty.minor == 1)
        {
            totalCudaCores = (deviceProperty.multiProcessorCount*48);
            printf("Total CUDA cores: %d \n", totalCudaCores);
        }
        else
        {
            printf("Total CUDA cores unknown, version 2.");
            printf("%d was released after this software was written.\n", deviceProperty.minor);
        }
    }
    else
    {
        printf("Total CUDA cores unknown, version %d.", deviceProperty.major);
        printf("%d was released after this software was written.\n", deviceProperty.minor);
    }

    

}

// --------------------------------------------------------------------------------
// Main function
// --------------------------------------------------------------------------------


void your_histogram_and_prefixsum(const float * const d_pArrLogLuminance,
                                  unsigned int * const d_pArrCDF,
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

    ShowDeviceProperty();

    // display picture info
    printf("pNumRows=%d\n", pNumRows);
    printf("pNumCols=%d\n", pNumCols);
    printf("pNumBins=%d\n", pNumBins);
    // dummy1<<<1, 1>>>(d_logLuminance);

    // Declarations
    unsigned int * vArrHistogram;
    
    // Calc the min and max value of d_logLuminance
    CalcMinMax(d_pArrLogLuminance,
               pNumRows,
               pNumCols,
               pMin_logLum,
               pMax_logLum);

    // Allocate memory for histogram
    checkCudaErrors(cudaMalloc(&vArrHistogram, sizeof(unsigned int) * pNumBins));

    // Generate histogram    
    /*
    CalcHistogramOld(d_pArrLogLuminance,
                     pNumRows,
                     pNumCols,
                     pNumBins,
                     pMin_logLum,
                     pMax_logLum,
                     vArrHistogram);
    */
    /*
    CalcHistogramReduce(d_pArrLogLuminance,
                        pNumRows,
                        pNumCols,
                        pNumBins,
                        pMin_logLum,
                        pMax_logLum,
                        vArrHistogram);
    */
    
    CalcHistogramNew(d_pArrLogLuminance,
                     pNumRows,
                     pNumCols,
                     pNumBins,
                     pMin_logLum,
                     pMax_logLum,
                     vArrHistogram);
    
    // Calculate the CDF
    CalcCDF(vArrHistogram,
            pNumBins,
            d_pArrCDF);

    // Free memory for histogram
    checkCudaErrors(cudaFree(vArrHistogram));

}