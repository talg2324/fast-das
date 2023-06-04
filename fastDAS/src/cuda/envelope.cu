#include <cstring>
#include "fft4g.cu"

__device__ void hilbert(double *signal, double *R, double *I, double *signal_ptr, int *ip, double *w, int N)
{
    int twoN = 2 * N;
    double normFactor = 2.0 / N;

    for (int t = 0; t < N; ++t)
    {
        signal_ptr[2 * t] = (double)signal[t]; // signalPtr[2 * t]     = Real(signal[t])
        signal_ptr[2 * t + 1] =             0;// signalPtr[2 * t + 1] = Imag(signal[t])
    }

    cdft(twoN, -1, signal_ptr, ip, w); // Forward FFT

    // Spectral Hilbert transform
    for (int t = 2; t < N; t += 2)
    {
        signal_ptr[t] *= 2;
        signal_ptr[t + 1] *= 2;
    }

    for (int t = N; t < twoN; ++t) signal_ptr[t] = 0;
    
    cdft(twoN, 1, signal_ptr, ip, w); // Inverse FFT
    
    double *pR = R;
    double *pI = I;

    for (int t = 0; t < twoN; t += 2)
    {
        *pR++ = signal_ptr[t] * normFactor;      // signalPtr[2 * t]     = Real(signal[t])
        *pI++ = signal_ptr[t + 1] * normFactor;  // signalPtr[2 * t + 1] = Imag(signal[t])
    }
}