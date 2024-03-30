# fast-DAS
Parallel delay-and-sum implementation wrapping native CUDA from python

<p align="center" width="100%">
    <img width="100%" src="./example/pw-sample.png"> 
</p>

## About
To take advantage of native code multi-threading libraries, use CalculateIn.CUDA or CalculateIn.CPP
- Supports windows and unix
- Cross-compiled on windows using WSL and CMake

Otherwise, you can use the slow python version with CalculateIn.PYTHON

## Usage
fast-DAS is a python wrapper and native code implementation for the delay-and-sum algorithm pre-compiled in C++ (CPU version) and CUDA (GPU version).  
The goal is to boost the digital beamforming process for ultrasound acquisitions.

### Installation

```console
pip install fast-das
```

### Note on Windows/CPU usage
In Windows, you may encounter the following error when trying to use the C++ code:

```console
FileNotFoundError: Could not find module '..\lib\site-packages\fastDAS\bin\libdas.dll' (or one of its dependencies). Try using the full path with constructor syntax.
```

This is because multi-threading is performed with the pthreads library, which is not natively included in Windows.
pthreads is included in MinGW, so adding gcc to your conda environment will resolve this issue (for example from [here](https://anaconda.org/conda-forge/m2w64-toolchain)).
### Input format
To beamform with fast-DAS, initialize a DAS object based on your compute preference (CPU/GPU):

```python
    import fastDAS as fd
    das = fd.DAS(target=fd.CalculateIn.CUDA)
    us_im = das.beamform(RF, del_Tx, del_Rx)
```

For <i>out-of-the-box</i> usage with Verasonics Vantage systems, save the MATLAB workspace after collecting an RF acquisition.  
The workspace file should contain at least the following variables as a .mat file:

```
    {
        'NA': {'NA_tot':_, 'NA':_},
        'PData': {
            'Size': _,
            'PDelta': _,
            'Origin': _,
        },
        'Trans': {
            'lensCorrection': _,
            'ElementPos': _,
            'WavelenToMm': _,
        },
        'TX': {
            'Steer': _,
            'Delay': _,
        },
        'TW': {
            'Peak': _
        },
        'Receive': {
            'SamplesPerWave': _,
            'StartDepth': _,
            'StartSample': _,
            'EndSample': _
        }
    }
```

In a plane wave compounding acqusition, the vsx.py is a basic example for how to calculate the Tx and Rx delay map based on a workspace like the one above.  
Otherwise, you can implement the ```del_Tx``` and ```del_Rx``` variables yourself.  
Note that the units of these delay maps are not ```[usec]``` but rather ```[samples]```.