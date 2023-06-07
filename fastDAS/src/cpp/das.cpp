#include <iostream>
#include <cmath>
#include <unordered_map>
#include <tuple>
#include "das.h"
#include "envelope.cpp"
#include <pthread.h>
#include <vector>
#include <thread>

/// @brief calculate the complex envelope of the RF data
/// @param RF input RF data
/// @param env_real real part of output envelope
/// @param env_imag imaginary part of output envelope
/// @param start_samp start index of the RF beam in the RF block
/// @param end_samp end index of the RF beam in the RF block
/// @param n_ang number of steering angles to beamform
/// @param n_el number of elements in the transducer aperture
/// @param N points per beam- should be a power of two
/// @param tot_samples number of samples in an RF block for a single angle
void envelope(double *RF, double *env_real, double *env_imag,
              short *start_samp, short *end_samp, int n_ang, int n_el, int N, int tot_samples)
{
    
    std::vector<pthread_t> threads;
    size_t n_threads = std::thread::hardware_concurrency();

    threads.reserve(n_threads);  // Reserve space for the threads

    int tasks_per_core = n_el / n_threads;
    int points_per_ang = n_el * N;
    int thread_idx = 0;

    for (int el = 0; el < n_el; el += tasks_per_core)
    {
        // Create thread arguments
        EnvelopeThreadArgs* args = new EnvelopeThreadArgs{
            RF, env_real, env_imag, start_samp, end_samp,
            n_ang, N, tot_samples, points_per_ang, el, el + tasks_per_core
        };

        pthread_t thread;
        pthread_create(&thread, nullptr, envelope_thread, args);
        threads.emplace_back(thread);
    }
    
    // Wait for all threads to finish
    for (pthread_t thread : threads) {
        pthread_join(thread, nullptr);
    }
}

/// @brief a logic block to be split among compute resources for envelope calculation
/// @return nullptr indicating thread completion
void* envelope_thread(void *args)
{
    EnvelopeThreadArgs *threadArgs = (EnvelopeThreadArgs*)args;

    double *RF = threadArgs->RF;
    double *env_real = threadArgs->env_real;
    double *env_imag = threadArgs->env_imag;
    short *start_samp = threadArgs->start_samp;
    short *end_samp = threadArgs->end_samp;
    int n_ang = threadArgs->n_ang;
    int N = threadArgs->N;
    int tot_samples = threadArgs->tot_samples;
    int points_per_ang = threadArgs->points_per_ang;
    int start_el = threadArgs->start_el;
    int end_el = threadArgs->end_el;

    int src_cursor, dst_cursor;
    int start;
    
    int sqrtN = ceil(sqrt(N));
    double *signal_ptr = (double*) malloc(sizeof(double) * 2 * N);
    int *ip = (int*) malloc(sizeof(int) * (sqrtN + 2));
    double *w = (double*) malloc(sizeof(double) * (N * 5) / 4);
    ip[0] = 0;

    src_cursor = start_el * tot_samples;

    for (int el = start_el; el < end_el; ++el)
    {
        dst_cursor = el * N;
        for (int n = 0; n < n_ang; ++n)
        {
            start = start_samp[n];

            hilbert(&RF[src_cursor + start], &env_real[dst_cursor], &env_imag[dst_cursor], signal_ptr, ip, w, N);

            dst_cursor += points_per_ang;
        }
        src_cursor += tot_samples;
    }

    free(signal_ptr);
    free(ip);
    free(w);
    return nullptr;
}

/// @brief delay and sum algorithm
/// @param us_im_real real part of output image
/// @param us_im_imag imaginary part of output image
/// @param env_real real part of complex envelope calculated from the envelope() function
/// @param env_imag imaginary part of complex envelope calculated from the envelope() function
/// @param delay_tx Tx delays in units of [samples] for each acquisition
/// @param delay_rx Rx delays in units of [samples] for each element
/// @param n_ang number of steering angles to beamform
/// @param n_el number of elements in the transducer aperture
/// @param N points per beam- should be a power of two
/// @param height grid height- same size as Tx and Rx delays
/// @param width grid with- same size as Tx and Rx delays
void delay_and_sum(double *us_im_real, double *us_im_imag, double *env_real, double *env_imag,
                   double *delay_tx, double *delay_rx, int n_ang, int n_el, int N, int height, int width)
{
    int points_per_ang = n_el * N;
    double *sample_vector = (double*) malloc(sizeof(double) * N);
    for (int i = 0; i < N; ++i) sample_vector[i] = i + 0.5;
    
    std::vector<pthread_t> threads;
    int n_threads = n_ang;

    threads.reserve(n_threads);  // Reserve space for the threads
    
    int pixels_per_slice = height * width;
    int n_pixels = n_ang * pixels_per_slice;

    int pixels_per_core = n_pixels / n_threads;
    for (int t = 0; t < n_threads; ++t)
    {
        // Create thread arguments

        InterpolationThreadArgs* args = new InterpolationThreadArgs{
            us_im_real, us_im_imag, env_real, env_imag,
            delay_tx, delay_rx, sample_vector, n_el, N, t,
            pixels_per_core, pixels_per_slice
        };

        pthread_t thread;
        pthread_create(&thread, nullptr, interp_field, args);
        threads.emplace_back(thread);
    }

    // Wait for all threads to finish
    for (pthread_t thread : threads) {
        pthread_join(thread, nullptr);
    }

    free(sample_vector);
}

/// @brief a logic block performing the interpolation step- uses the Tx and Rx delays to create an image from each element RF
/// @return nullptr indicating thread completion
void* interp_field(void *args)
{
    InterpolationThreadArgs *threadArgs = (InterpolationThreadArgs*)args;

    double *us_im_real = threadArgs->us_im_real;
    double *us_im_imag = threadArgs->us_im_imag;
    double *env_real = threadArgs->env_real;
    double *env_imag = threadArgs->env_imag;
    double *delay_tx = threadArgs->delay_tx;
    double *delay_rx = threadArgs->delay_rx;
    double *sample_vector = threadArgs->sample_vector;
    int N = threadArgs->N;
    int n_el = threadArgs->n_el;
    int thread_idx = threadArgs->thread_idx;
    int pixels_per_core = threadArgs->pixels_per_core;
    int pixels_per_slice = threadArgs->pixels_per_slice;
    
    int volume_offset = thread_idx * pixels_per_core;
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

        for (int px = 0; px < pixels_per_core; ++px)
        {
            delay_samples = del_tx[px] + del_rx[px];
            if (delay_samples > sample_vector[N-2] || delay_samples < sample_vector[1]) continue;
            else {

                // Find the nearest sampled point
                // t = binary_search_nearest(sample_vector, N, delay_samples);
                t = (int)round(delay_samples);

                denominator = sample_vector[t] - sample_vector[t-1];
                w = (delay_samples - sample_vector[t-1]) / denominator;
                w2 = w * w;
                w3 = w * w2;
                
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
    
    delete threadArgs;
    pthread_exit(nullptr);
    return nullptr;
}

int binary_search_nearest(const double *arr, int size, double target) {
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