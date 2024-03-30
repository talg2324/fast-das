import numpy as np
from scipy.io import loadmat


def from_verasonics(data_dir, n_el=128, start_el=64):
    # Organize the data
    ws = load_workspace(data_dir + 'Workspace.mat')
    RF = load_rf(data_dir + 'RFData.mat')
    del_Tx, del_Rx = beamforming_pw_delays(ws, n_el, start_el)
    el_order = ws['Trans']['ConnectorES'][start_el:start_el+n_el].squeeze()-1
    end_sample = ws['Receive']['EndSample'].squeeze().astype(np.int16)
    na = del_Tx.shape[0]

    RF = RF[:end_sample[-1], el_order].astype(np.float64).T
    RF = RF.reshape(n_el, na, -1)
    return RF, del_Tx, del_Rx


def beamforming_pw_delays(ws, n_el, start_el):    
    im_res = (ws['PData']['PDelta'][-1], ws['PData']['PDelta'][0])  # x y z -> z x
    im_shape = tuple(ws['PData']['Size'][:-1])  # z x

    pix_pos = np.zeros((im_shape[0], im_shape[1], 3))

    scan_origin = np.arange(im_res[1], im_shape[1] + im_res[1]) * im_res[1]
    scan_origin += ws['PData']['Origin'][0]
    scan_origin = np.stack((scan_origin, 0*scan_origin, 0*scan_origin + ws['PData']['Origin'][2]), -1)

    scan_dir = np.zeros((im_shape[1], 2))
    scan_depth = np.reshape(np.arange(im_res[0], im_shape[0]+im_res[0]) * im_res[0], (im_shape[0], 1))

    pix_pos[..., 0] = np.ones((im_shape[0], 1)) * scan_origin[:, 0] + scan_depth * np.sin(scan_dir[:, 0]) * np.cos(scan_dir[:, 1])
    pix_pos[..., 2] = np.ones((im_shape[0], 1)) * scan_origin[:, 2] + scan_depth * np.cos(scan_dir[:, 1])

    el_posx = ws['Trans']['ElementPos'][start_el:start_el+n_el, 0] / ws['Trans']['WavelenToMm']  # [wavelengths]
    el_posz = ws['Trans']['ElementPos'][start_el:start_el+n_el, 2] / ws['Trans']['WavelenToMm']  # [wavelengths]
    ws['Trans']['lensCorrection'] /= ws['Trans']['WavelenToMm']

    x = pix_pos[..., 0]  # [wavelengths]
    z = pix_pos[..., 2]  # [wavelengths]

    # Rx Focus
    # Rx delay is independent of transmission
    del_Rx = np.zeros((n_el, im_shape[0], im_shape[1]))

    Rx_const_delay = ws['Trans']['lensCorrection'] - ws['Receive']['StartDepth']

    for n in range(n_el):
        del_Rx[n, ...] = np.hypot(x-el_posx[n], z-el_posz[n])

    del_Rx = (del_Rx + Rx_const_delay) * ws['Receive']['SamplesPerWave']

    # TX Focus
    NA = ws['NA']['NA_tot'][0, 0]
    del_Tx = np.zeros((NA, im_shape[0], im_shape[1]))
    for n in range(NA):
        phi = ws['TX']['Steer'][n]

        del_Tx[n, ...] = z * np.cos(phi) + x * np.sin(phi)
        del_Tx[n, ...] += np.interp(0, el_posx, ws['TX']['Delay'][n])  # Add delay to origin

    Tx_const_delay = ws['TW']['Peak'] + ws['Trans']['lensCorrection'] - ws['Receive']['StartDepth']

    del_Tx = (del_Tx + Tx_const_delay) * ws['Receive']['SamplesPerWave']
    return del_Tx, del_Rx


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
