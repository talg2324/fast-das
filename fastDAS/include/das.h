#ifndef MYHEADER_H
#define MYHEADER_H

#ifdef __cplusplus
extern "C" {
#endif

void envelope(double *RF, double *env_real, double *env_imag, int *start_samp, int n_ang, int n_el, int N, int tot_samples);

void delay_and_sum(double *us_im_real, double *us_im_imag,
                   double *env_real, double *env_imag,
                   double *delay_tx, double *delay_rx,
                   int n_ang, int n_el, int N, int height, int width);

#ifdef __cplusplus
}
#endif

#endif

void* envelope_thread(void *args);
void* interp_field(void *args);

int binary_search_nearest(const double *arr, int size, double target);

struct EnvelopeThreadArgs {
    double *RF;
    double *env_real;
    double *env_imag;
    int *start_samp;
    int n_ang;
    int N;
    int tot_samples;
    int points_per_ang;
    int start_el;
    int end_el;
};

struct InterpolationThreadArgs {
    double *us_im_real;
    double *us_im_imag;
    double *env_real;
    double *env_imag;
    double *delay_tx;
    double *delay_rx;
    double *sample_vector;
    int n_el;
    int N;
    int thread_idx;
    int pixels_per_core;
    int pixels_per_slice;
};