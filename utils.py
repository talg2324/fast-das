from scipy.io import loadmat
import numpy as np


def load_rf(f):
    return np.squeeze(loadmat(f)['RFData_tot']).astype(np.int16)


def load_workspace(f):
    WS = loadmat(f)
    WS = fix_ws(WS)

    return WS


def fix_ws(WS):
    return {
        'NA': {'NA_tot': WS['na_tot'], 'NA': WS['na']},
        'PData': {
            'Size': WS['PData']['Size'][0][0].flatten(),
            'PDelta': WS['PData']['PDelta'][0][0].flatten(),
            'Origin': WS['PData']['Origin'][0][0].flatten(),
        },
        'Trans': {
            'lensCorrection': WS['Trans']['lensCorrection'][0][0].flatten(),
            'ElementPos': WS['Trans']['ElementPos'][0][0],
            'WavelenToMm': WS['Trans']['spacingMm'][0][0].flatten() / WS['Trans']['spacing'][0][0].flatten(),
            'ConnectorES': WS['Trans']['ConnectorES'][0][0]
        },
        'TX': {
            'Steer': np.array([WS['TX']['Steer'][0][i][0, 0] for i in range(WS['TX'].shape[-1])]),
            'Delay': np.vstack([WS['TX']['Delay'][0, i] for i in range(9)]),
            'TXPD': [WS['TX'][0, i][-1][..., 0] > 1e3 for i in range(WS['TX'].shape[1])]
        },
        'TW': {
            'Peak': WS['TW']['peak'][0][0].flatten()
        },
        'Receive': {
            'SamplesPerWave': WS['Receive']['samplesPerWave'][0][0].flatten(),
            'StartDepth': WS['Receive']['startDepth'][0][0].flatten(),
            'StartSample': np.vstack([WS['Receive']['startSample'][0][i].flatten() for i in range(WS['Receive']['endSample'].shape[1])]),
            'EndSample': np.vstack([WS['Receive']['endSample'][0][i].flatten() for i in range(WS['Receive']['endSample'].shape[1])])
        }
    }
