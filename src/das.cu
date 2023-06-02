#include <iostream>
#include <cmath>
#include <unordered_map>
#include <tuple>
#include "das.cuh"
#include "envelope.cu"
#include <vector>
#include <thread>

void cuda_valid(bool *res)
{
    int n_devices;

    cudaGetDeviceCount(&n_devices);
    cudaError_t error = cudaGetLastError();

    if (n_devices == 0 || error != cudaSuccess) 
    {
        *res = false;
    } else {
        *res = true;
    }

}

void envelope(double *RF, double *env_real, double *env_imag,
              short *start_samp, short *end_samp, int n_ang, int n_el, int N, int tot_samples)
{
    double *RF_cuda, *env_real_cuda, *env_imag_cuda;
    short *start_samp_cuda, *end_samp_cuda;

    cudaMalloc((void**)&RF_cuda,         sizeof(double) * n_el * tot_samples);
    cudaMalloc((void**)&env_real_cuda,   sizeof(double) * n_ang * n_el * N);
    cudaMalloc((void**)&env_imag_cuda,   sizeof(double) * n_ang * n_el * N);
    cudaMalloc((void**)&start_samp_cuda, sizeof(short)  * n_ang);
    cudaMalloc((void**)&end_samp_cuda,   sizeof(short)  * n_ang);

    cudaMemcpyAsync(RF_cuda,                 RF, sizeof(double) * n_el * tot_samples, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(env_real_cuda,     env_real, sizeof(double) * n_ang * n_el * N, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(env_imag_cuda,     env_imag, sizeof(double) * n_ang * n_el * N, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(start_samp_cuda, start_samp, sizeof(short)  * n_ang, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(end_samp_cuda,     end_samp, sizeof(short)  * n_ang, cudaMemcpyHostToDevice);

    // Helper arrays
    double *signal_ptr, *w;
    int *ip;
    int sqrtN = ceilf(sqrtf(N));

    cudaMalloc((void**)&signal_ptr , sizeof(double) * n_el * 2 * N);
    cudaMalloc((void**)&ip,          sizeof(int) * n_el * (sqrtN + 2));
    cudaMalloc((void**)&w,           sizeof(double) * n_el * (N * 5) / 4);

    int threads_per_block = n_el / GRID_SIZE_ENV;
    envelope_thread<<<GRID_SIZE_ENV, threads_per_block>>>(RF_cuda, env_real_cuda, env_imag_cuda, 
                                                        start_samp_cuda, end_samp_cuda,
                                                        signal_ptr, ip, w,
                                                        n_ang, n_el, N, tot_samples);
    cudaDeviceSynchronize();
    cudaFree(signal_ptr);
    cudaFree(ip);
    cudaFree(w);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }

    cudaMemcpyAsync(env_real, env_real_cuda, sizeof(double) * n_ang * n_el * N, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(env_imag, env_imag_cuda, sizeof(double) * n_ang * n_el * N, cudaMemcpyDeviceToHost);
    cudaFree(RF_cuda);
    cudaFree(env_real_cuda);
    cudaFree(env_imag_cuda);
    cudaFree(start_samp_cuda);
    cudaFree(end_samp_cuda);
}

__global__ void envelope_thread(double *RF, double *env_real, double *env_imag,
              short *start_samp, short *end_samp, double *signal_ptr, int *ip, double *w,
              int n_ang, int n_el, int N, int tot_samples)
{
    int el = blockIdx.x * blockDim.x + threadIdx.x;

    if (el < n_el)
    {
        int sqrtN = sqrtf(N);
        int fft_cursor = el * 2 * N;
        int ip_cursor = el * (sqrtN + 2);
        int w_cursor = el * (N * 5) / 4;

        ip[ip_cursor] = 0;

        int src_cursor, dst_cursor, start;
        src_cursor = el * tot_samples;

        int points_per_ang = n_el * N;

        for (int n = 0; n < n_ang; ++n)
        {
            start = start_samp[n];
            dst_cursor = n * points_per_ang + el * N;
            hilbert(&RF[src_cursor + start], &env_real[dst_cursor], &env_imag[dst_cursor],
                    &signal_ptr[fft_cursor], &ip[ip_cursor], &w[w_cursor], N);
        }
    }
}


void delay_and_sum(double *us_im_real, double *us_im_imag, double *env_real, double *env_imag,
                   double *delay_tx, double *delay_rx, int n_ang, int n_el, int N, int height, int width)
{

    double *us_im_real_cuda, *us_im_imag_cuda, *env_real_cuda, *env_imag_cuda;
    double *delay_tx_cuda, *delay_rx_cuda;

    
    int pixels_per_slice = height * width;
    int n_pixels = n_ang * pixels_per_slice;
    int RF_samples = n_ang * n_el * N;
    int pixels_per_thread = n_pixels / GRID_SIZE_INTERP / MAX_THREADS;

    cudaMalloc((void**)&us_im_real_cuda, sizeof(double) * n_pixels);
    cudaMalloc((void**)&us_im_imag_cuda, sizeof(double) * n_pixels);
    cudaMalloc((void**)&env_real_cuda,   sizeof(double) * RF_samples);
    cudaMalloc((void**)&env_imag_cuda,   sizeof(double) * RF_samples);
    cudaMalloc((void**)&delay_tx_cuda,   sizeof(double) * n_pixels);
    cudaMalloc((void**)&delay_rx_cuda,   sizeof(double) * n_el * pixels_per_slice);

    cudaMemcpyAsync(us_im_real_cuda, us_im_real, sizeof(double) * n_pixels, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(us_im_imag_cuda, us_im_imag, sizeof(double) * n_pixels, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(env_real_cuda,   env_real,   sizeof(double) * RF_samples, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(env_imag_cuda,   env_imag,   sizeof(double) * RF_samples, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(delay_tx_cuda,   delay_tx,   sizeof(double) * n_pixels, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(delay_rx_cuda,   delay_rx,   sizeof(double) * n_el  * pixels_per_slice, cudaMemcpyHostToDevice);

    double *sample_vector, *sample_vector_cuda;

    sample_vector = (double*) malloc(sizeof(double) * N);
    for (int i = 0; i < N; ++i) sample_vector[i] = i + 0.5;

    cudaMalloc((void**)&sample_vector_cuda, sizeof(double) * N);
    cudaMemcpyAsync(sample_vector_cuda, sample_vector, sizeof(double) * N, cudaMemcpyHostToDevice);
    free(sample_vector);
    
    interp_field_thread<<<GRID_SIZE_INTERP, MAX_THREADS>>>(us_im_real_cuda, us_im_imag_cuda, env_real_cuda, env_imag_cuda,
                                                          delay_tx_cuda, delay_rx_cuda, sample_vector_cuda,
                                                          n_ang, n_el, N, pixels_per_slice, pixels_per_thread);
    cudaDeviceSynchronize();
    cudaMemcpyAsync(us_im_real, us_im_real_cuda, sizeof(double) * n_pixels, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(us_im_imag, us_im_imag_cuda, sizeof(double) * n_pixels, cudaMemcpyDeviceToHost);

    cudaFree(us_im_real_cuda);
    cudaFree(us_im_imag_cuda);
    cudaFree(env_real_cuda);
    cudaFree(env_imag_cuda);
    cudaFree(delay_tx_cuda);
    cudaFree(delay_rx_cuda);
    cudaFree(sample_vector_cuda);
}

__global__ void interp_field_thread(double *us_im_real, double *us_im_imag, double *env_real, double *env_imag,
                                    double *delay_tx, double *delay_rx, double *sample_vector, int n_ang, int n_el, int N, int pixels_per_slice, int pixels_per_thread)
{

    int volume_offset = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int n = volume_offset / pixels_per_slice;
    int slice_offset = volume_offset - n * pixels_per_slice;

    int RF_offset = n * n_el * N;

    double *beam_real, *beam_imag, *del_rx, *del_tx, *dst_real, *dst_imag;
    double fx0, fx1, fx2, fx3;
    double a0, a1, a2, a3;
    double delay_samples, denominator, w, w2, w3;
    int t;

    for (int el = 0; el < n_el; ++el)
    {        
        beam_real = &env_real[RF_offset + el * N];
        beam_imag = &env_imag[RF_offset + el * N];

        del_tx = &delay_tx[volume_offset];
        del_rx = &delay_rx[slice_offset + el * pixels_per_slice];

        dst_real = &us_im_real[volume_offset];
        dst_imag = &us_im_imag[volume_offset];

        for (int px = 0; px < pixels_per_thread; ++px)
        {
            delay_samples = del_tx[px] + del_rx[px];
            if (delay_samples > sample_vector[N-2] || delay_samples < sample_vector[1]) continue;
            else {

                // Find the nearest sampled point
                t = binary_search_nearest(sample_vector, N, delay_samples);

                if (t == -1) {
                    continue;
                } else {

                    denominator = sample_vector[t] - sample_vector[t-1];
                    w = (delay_samples - sample_vector[t-1]) / denominator;
                    w2 = w * w;
                    w3 = w * w2;
                }
                
                fx0 = beam_real[t - 2];
                fx1 = beam_real[t - 1];
                fx2 = beam_real[t    ];
                fx3 = beam_real[t + 1];

                a0 = -0.5 * fx0 + 1.5 * fx1 - 1.5 * fx2 + 0.5 * fx3;
                a1 = fx0 - 2.5 * fx1 + 2.0 * fx2 - 0.5 * fx3;
                a2 = -0.5 * fx0 + 0.5 * fx2;
                a3 = fx1;

                dst_real[px] += a0 * w3 + a1 * w2 + a2 * w + a3;

                fx0 = beam_imag[t - 2];
                fx1 = beam_imag[t - 1];
                fx2 = beam_imag[t    ];
                fx3 = beam_imag[t + 1];
                
                a0 = -0.5 * fx0 + 1.5 * fx1 - 1.5 * fx2 + 0.5 * fx3;
                a1 = fx0 - 2.5 * fx1 + 2.0 * fx2 - 0.5 * fx3;
                a2 = -0.5 * fx0 + 0.5 * fx2;
                a3 = fx1;

                dst_imag[px] += a0 * w3 + a1 * w2 + a2 * w + a3;     
            }
        }
    }
}

__device__ int binary_search_nearest(const double *arr, int size, double target) {
    int left = 0;
    int right = size - 1;

    int nearestIndex = -2;
    double minDiff = size;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        double diff = target - arr[mid];

        if (diff < minDiff && diff > 0) {
            minDiff = diff;
            nearestIndex = mid + 1;
        }
        if (arr[mid] < target) {
            left = mid + 1;  // Search in the right half
        } else {
            right = mid - 1;  // Search in the left half
        }
    }
    
    return nearestIndex;  // Return index of the nearest value
}