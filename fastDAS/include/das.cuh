#ifndef MYHEADER_H
#define MYHEADER_H

#define GRID_SIZE_ENV 1
#define GRID_SIZE_INTERP 512
#define MAX_THREADS 1024

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define MODULE_API __declspec(dllexport)
#else
#define MODULE_API
#endif

    MODULE_API void cuda_valid(bool *res);

    MODULE_API void envelope(double *RF, double *env_real, double *env_imag, int *start_samp, int n_ang, int n_el, int N, int tot_samples);

    MODULE_API void delay_and_sum(double *us_im_real, double *us_im_imag,
                                         double *env_real, double *env_imag,
                                         double *delay_tx, double *delay_rx,
                                         int n_ang, int n_el, int N, int pixels_per_slice, int pixels_per_thread);

#ifdef __cplusplus
}
#endif

#endif

__global__ void envelope_thread(double *RF, double *env_real, double *env_imag,
              int *start_samp, double *signal_ptr, int *ip, double *w,
              int n_ang, int n_el, int N, int tot_samples);
              
__global__ void interp_field_thread(double *us_im_real, double *us_im_imag, double *env_real, double *env_imag,
                                    double *delay_tx, double *delay_rx, double *sample_vector, int n_ang, int n_el, int N, int pixels_per_slice, int pixels_per_thread);

__device__ int binary_search_nearest(const double *arr, int size, double target);