## PCI ST-Library. By Adenauer Casali.

# started: 26/10/2017
# last update: 01/08/2018

import numpy as np
from numpy import linalg
import scipy.signal
from sklearn.decomposition import PCA
import time
import datetime
from bitarray import bitarray
from ipywidgets import FloatProgress
from IPython.display import display
import pickle
import scipy.io as sio
import os.path
import pandas as pd


##MAIN:
def calc_PCIst(signal_evk, times, **par):
    signal_evk, times = preprocess_signal(signal_evk, times, (par['baseline_window'][0], par['response_window'][1]),
                                          **par)
    signal_svd, eigenvalues = dimensionality_reduction(signal_evk, times, par['response_window'], **par)
    results = state_transition_quantification(signal_svd, times, **par)
    return results


##BATCH (tables, files,...):
def run_session(data_evk, ix, **par):
    times = data_evk[ix]['times']
    signal_evk = data_evk[ix]['EVK']
    results = calc_PCIst(signal_evk, times, **par)
    return results


def run_batch(data_evk, sessions, variables=['PCI', 'PCI_bydim'], func=None, **par):
    # assert len(data_evk)==len(sessions), 'data and sessions must map onto each other.'
    ini = time.time()
    n_sessions = len(sessions)
    progress_bar = FloatProgress(min=0, max=n_sessions, description='Calculating...')
    display(progress_bar)

    batch_data = []
    for ix in sessions:
        ini2 = time.time()
        if not func:
            results = run_session(data_evk, ix, **par)
        else:
            results = func(data_evk, ix, **par)
        end2 = time.time()
        session_time = end2 - ini2

        if 'elapsed_time' in variables:
            results['elapsed_time'] = session_time
        results_reduced = {key: results[key] for key in variables}

        batch_data.append(results_reduced)
        progress_bar.description = '{}/{}'.format(int(progress_bar.value), n_sessions - 1)
        progress_bar.value = progress_bar.value + 1

    progress_bar.description = 'Done.'
    progress_bar.close()
    print('Data successfully calculated.')
    end = time.time()
    print('Elapsed time: {:.2f}s'.format(end - ini))
    return batch_data


def save2pickle(obj, filename, path=None, date_stamp=True):
    if not path:
        path = '/Users/gryllos/Documents/Pesquisa/Mestrado/Novo PCI/'
    if date_stamp:
        year = str(datetime.datetime.today().year)
        month = str(datetime.datetime.today().month)
        day = str(datetime.datetime.today().day)
        filename = '_'.join([filename, year, month, day]) + '.p'
    else:
        filename = filename + '.p'
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_dataset(table, dataset_name, base_path):
    data_evk = {}
    for ix in table.index.values:
        condition = table.loc[ix, 'Condition']
        file_name = table.loc[ix, 'File']
        file_path = base_path / dataset_name / condition / file_name
        if file_path.exists():
            with file_path.open('rb') as f:
                datafile = pickle.load(f)
        else:
            print('Error loading {}/{}'.format(condition, file_name))
            print(file_path)
            datafile = []
        data_evk[ix] = datafile
    return data_evk


def ys2table(ys, table, sessions=None, y_label='PCI'):
    if not sessions:
        sessions = list(table.index.values)
    assert len(ys) == len(sessions), 'ys and sessions must map onto each other.'
    for i in range(len(ys)):
        ix = sessions[i]
        table.loc[ix, y_label] = ys[i]
    return table


def save_table(table, table_name, file_path=''):
    table_format = table_name.split('.')[-1]
    table_path = os.path.join(file_path, table_name)
    if table_format == 'xlsx':
        writer = writer = pd.ExcelWriter(table_path, engine='xlsxwriter')
        table.to_excel(writer)
    elif table_format == 'csv':
        table.to_csv(table_path)
    else:
        print('Table format not accepted.')


def get_par_id(par, name='PCIst'):
    s = [name, par['baseline_window'][0], par['baseline_window'][1], par['response_window'][0],
         par['response_window'][1],
         'k', round(par['k'], 1), 'maxVar', par['max_var'], 'nsteps', par['n_steps'], 'min_snr',
         round(par['min_snr'], 1)]
    s = [str(x) for x in s]
    if par['embed']:
        embedding = '_'.join(['m', str(par['m']), 'tau', str(par['tau'])])
        s.append(embedding)
    else:
        s.append('m_1')
    s = '_'.join(s)
    return s


## DIMENSIONALITY REDUCTION
def dimensionality_reduction(signal, times, svd_window, max_var, min_snr=None, n_components=None, **kwargs):
    '''Returns principal components of signal according to SVD of the response.

    Calculates SVD at a given time interval (t>0) and uses the new basis to transform the whole signal yielding `n_components` principal components.
    The principal components are then selected to account for at least `max_var`% of the variance basesent in the signal's response.

    Parameters
    ----------
    signal : ndarray
        2D array (ch,time) containing signal.
    times : ndarray
        1D array (time,) containing timepoints (negative values are baseline)
    svd_window : tuple
        Tuple (t_ini,t_end) with time interval of the response on which SVD is calculated.
    max_var: 0 < float <= 100
        Percentage of variance accounted for by the selected principal components.
    n_components : int, optional
        Number of principal components calculated (before selection).


    Returns
    -------
    np.ndarray
        2D array (ch,time) with selected principal components.
    np.ndarray
        1D array (n_components,) with `n_components` SVD eigenvalues of the signal's response.
    '''

    if not n_components:
        n_components = signal.shape[0]

    Vk, eigenvalues = get_svd(signal, times, svd_window, n_components)

    signal_svd = signal.T.dot(Vk).T

    max_dim = calc_maxdim(eigenvalues, max_var)

    signal_svd = signal_svd[:max_dim, :]

    if min_snr:
        base_ini_ix, base_end_ix = get_time_index(times, kwargs['baseline_window'][0]), get_time_index(times, kwargs[
            'baseline_window'][1])
        resp_ini_ix, resp_end_ix = get_time_index(times, kwargs['response_window'][0]), get_time_index(times, kwargs[
            'response_window'][1])
        n_dims = np.size(signal_svd, 0)
        snrs = np.zeros(n_dims)
        for c in range(n_dims):
            snrs[c] = np.sqrt((np.divide(np.mean(np.square(signal_svd[c, resp_ini_ix:resp_end_ix])),
                                         np.mean(np.square(signal_svd[c, base_ini_ix:base_end_ix])))))
        signal_svd = signal_svd[snrs > min_snr, :]

    vk = Vk[:, :signal_svd.shape[0]].T  # get loads (components x channels) todo: confirm
    return signal_svd, eigenvalues, vk


def get_svd(signal_evk, times, svd_window, n_components):
    ini_t, end_t = svd_window
    ini_ix = get_time_index(times, onset=ini_t)
    end_ix = get_time_index(times, onset=end_t)

    signal_resp = signal_evk[:, ini_ix:end_ix].T  # we want the matrix samples x dimensions, that is, time x channels

    U, S, V = linalg.svd(signal_resp, full_matrices=False)
    V = V.T
    Vk = V[:, :n_components]
    eigenvalues = S[:n_components]

    return Vk, eigenvalues


## STATE TRANSITION QUANTIFICATION
def state_transition_quantification(signal, times, k, baseline_window, response_window, embed=False, m=None, tau=None,
                                    n_steps=50, max_thr_p=1.0, **kwargs):
    ''' Receives selected principal components of perturbational signal and performs state transition quantification.

    Parameters
    ----------
    signal : ndarray
        2D array (component,time) containing signal (typically, the selected principal components).
    times : ndarray
        1D array (time,) containing timepoints (negative values are baseline).
    k : float > 1
        Noise control parameter.
    baseline_window : tuple
        Signal's baseline time interval (ini,end).
    response_window : tuple
        Signal's response time interval (ini,end).
    embed : bool, optional
        Perform time-delay embedding.
    m : int
        Number of embedding dimensions.
    tau : int
        Number of timesamples of embedding delay
    n_steps : int, optional

    Returns
    -------
    The output is a dictionary with the following fields:
    PCI : float
    PCI_bydim : ndarray
    D_base : ndarray
        Distance matrix of the baseline for all components
    D_resp : ndarray
        Distance matrix of the response for all components
    T_base : ndarray
        Transition matrix of the baseline for all components
    T_resp : ndarray
        Transition matrix of the response for all components
    '''

    n_dims = signal.shape[0]
    if n_dims == 0:
        print('No components --> PCIst=0')
        results = {'PCI': 0, 'PCI_bydim': 0, 'D_base': 0, 'D_resp': 0, 'T_base': 0, 'T_resp': 0, 'n_dims': 0,
                   'thresholds': 0, 'NST_diff': 0, 'NST_resp': 0, 'NST_base': 0, 'max_thresholds': 0}
        return results

    # EMBEDDING
    if embed:
        cut = (m - 1) * tau
        times = times[cut:]
        temp_signal = np.zeros((n_dims, m, len(times)))
        for i in range(n_dims):
            temp_signal[i, :, :] = dimension_embedding(signal[i, :], m, tau)
        signal = temp_signal

    else:
        signal = signal[:, np.newaxis, :]

    # BASELINE AND RESPONSE DEFINITION
    base_ini_ix, base_end_ix = get_time_index(times, baseline_window[0]), get_time_index(times, baseline_window[1])
    resp_ini_ix, resp_end_ix = get_time_index(times, response_window[0]), get_time_index(times, response_window[1])

    n_baseline, n_response = len(times[base_ini_ix:base_end_ix]), len(times[resp_ini_ix:resp_end_ix])

    if n_response <= 1 or n_baseline <= 1:
        print('Warning: Bad time interval defined.')

    baseline = signal[:, :, base_ini_ix:base_end_ix]
    response = signal[:, :, resp_ini_ix:resp_end_ix]

    # NST CALCULATION

    # Distance matrix
    D_base = np.zeros((n_dims, n_baseline, n_baseline))
    D_resp = np.zeros((n_dims, n_response, n_response))
    # Transition matrix
    T_base = np.zeros((n_steps, n_dims, n_baseline, n_baseline))
    T_resp = np.zeros((n_steps, n_dims, n_response, n_response))

    # Number of mean state transitions
    NST_base = np.zeros((n_steps, n_dims))
    NST_resp = np.zeros((n_steps, n_dims))

    thresholds = np.zeros((n_steps, n_dims))

    for i in range(n_dims):
        D_base[i, :, :] = recurrence_matrix(baseline[i, :, :], thr=None, mode='distance')
        D_resp[i, :, :] = recurrence_matrix(response[i, :, :], thr=None, mode='distance')

        min_thr = np.median(D_base[i, :, :].flatten())
        # max_thr = np.max(D_base[i,:,:].flatten()) * 2
        # min_thr = 0
        max_thr = np.max(D_resp[i, :, :].flatten()) * max_thr_p
        # # max_thr = np.max(D_base[i,:,:].flatten())
        # max_thr = np.max([1,2,3])
        thresholds[:, i] = np.linspace(min_thr, max_thr, n_steps)

    for i in range(n_steps):
        for j in range(n_dims):
            T_base[i, j, :, :] = distance2transition(D_base[j, :, :], thresholds[i, j])
            T_resp[i, j, :, :] = distance2transition(D_resp[j, :, :], thresholds[i, j])

            NST_base[i, j] = np.sum(T_base[i, j, :, :]) / n_baseline ** 2
            NST_resp[i, j] = np.sum(T_resp[i, j, :, :]) / n_response ** 2

    NST_diff = NST_resp - k * NST_base
    ixs = np.argmax(NST_diff, axis=0)

    max_thresholds = np.array([thresholds[ix, i] for ix, i in zip(ixs, range(n_dims))])
    PCI_bydim = np.array([NST_diff[ix, i] for ix, i in zip(ixs, range(n_dims))]) * n_response
    PCI = np.sum(PCI_bydim)

    temp = np.zeros((n_dims, n_response, n_response))
    temp2 = np.zeros((n_dims, n_baseline, n_baseline))
    for i in range(n_dims):
        temp[i, :, :] = T_resp[ixs[i], i, :, :]
        temp2[i, :, :] = T_base[ixs[i], i, :, :]
    T_resp = temp
    T_base = temp2

    results = {'PCI': PCI, 'PCI_bydim': PCI_bydim, 'D_base': D_base, 'D_resp': D_resp, 'T_base': T_base,
               'T_resp': T_resp, 'n_dims': n_dims,
               'thresholds': thresholds, 'NST_diff': NST_diff, 'NST_resp': NST_resp, 'NST_base': NST_base,
               'max_thresholds': max_thresholds}
    return results


def recurrence_matrix(signal, mode, thr=None):
    ''' Calculates distance, recurrence or transition matrix. Signal can be embedded (m, n_times) or not (, n_times).

    Parameters
    ----------
    signal : ndarray
        Time-series; may be a 1D (time,) or a m-dimensional array (m, time) for time-delay embeddeding.
    mode : str
        Specifies calculated matrix: 'distance', 'recurrence' or 'transition'
    thr : float, optional
        If transition matrix is chosen (`mode`=='transition'), specifies threshold value.

    Returns
    -------
    ndarray
        2D array containing specified matrix.
    '''
    if len(signal.shape) == 1:
        signal = signal[np.newaxis, :]
    n_dims = signal.shape[0]
    n_times = signal.shape[1]

    R = np.zeros((n_dims, n_times, n_times))
    for i in range(n_dims):
        D = np.tile(signal[i, :], (n_times, 1))
        D = D - D.T
        R[i, :, :] = D
    R = np.linalg.norm(R, ord=2, axis=0)

    mask = (R <= thr) if thr else np.zeros(R.shape).astype(bool)
    if mode == 'distance':
        R[mask] = 0
        return R
    elif mode == 'recurrence':
        return mask.astype(int)
    elif mode == 'transition':
        return diff_matrix(mask.astype(int), symmetric=False)
    else:
        return 0


def distance2transition(distR, thr):
    ''' Receives 2D distance matrix and calculates transition matrix. '''
    mask = distR <= thr
    R = diff_matrix(mask.astype(int), symmetric=False)
    return R


def distance2recurrence(distR, thr):
    ''' Receives 2D distance matrix and calculates recurrence matrix. '''
    mask = distR <= thr
    return mask.astype(int)


def diff_matrix(A, symmetric=False):
    # grad_A = np.array(np.gradient(A))
    B = np.abs(np.diff(A))
    if B.shape[1] != B.shape[0]:
        B2 = np.zeros((B.shape[0], B.shape[1] + 1))
        B2[:, :-1] = B
        B = B2
    if symmetric:
        B = (B + B.T)
        B[B > 0] = 1
    return B


def calc_maxdim(eigenvalues, max_var):
    ''' Get number of dimensions that accumulates at least `max_var`% of total variance'''
    if max_var == 100:
        return len(eigenvalues)
    else:
        eigenvalues = np.sort(eigenvalues)[::-1]
        var = eigenvalues ** 2
        var_p = 100 * var / np.sum(var)
        var_cum = np.cumsum(var_p)
        max_dim = len(eigenvalues) - np.sum(var_cum >= max_var) + 1
        return max_dim


def dimension_embedding(x, m, tau):
    '''
    Returns time-delay embedding of vector.

    Parameters
    ----------
    x : ndarray
        1D array time series.
    m : int
        Number of dimensions in the embedding.
    tau : int
        Number of samples in delay.
    Returns
    -------
    ndarray
        2D array containing embedded signal (m, time)

    '''

    assert len(x.shape) == 1, "x must be one-dimensional array (n,)"
    n = x.shape[0]  # length of time-series
    s = np.zeros((m, n - (m - 1) * tau))
    ini = (m - 1) * tau if m > 1 else None
    s[0, :] = x[ini:]  # dimension "1"
    for i in range(1, m):
        ini = (m - i - 1) * tau
        end = -i * tau
        s[i, :] = x[ini:end]
    return s


## PREPROCESS

def preprocess_signal(signal_evk, times, time_window, baseline_corr=False, resample=None, **kwargs):
    assert signal_evk.shape[1] == len(times), 'Signal and Time arrays must be of the same size.'

    if 'avgref' in kwargs.keys():
        avgref = kwargs['avgref']
    else:
        avgref = None

    if avgref:
        signal_evk = rereference(signal_evk)

    if baseline_corr:
        signal_evk = baseline_correct(signal_evk, times, delta=-50)

    t_ini, t_end = time_window
    ini_ix = get_time_index(times, t_ini)
    end_ix = get_time_index(times, t_end)
    signal_evk = signal_evk[:, ini_ix:end_ix]
    times = times[ini_ix:end_ix]

    if resample:
        signal_evk, times = undersample_signal(signal_evk, times, new_fs=resample)

    return signal_evk, times


def undersample_signal(signal, times, new_fs):
    '''
    signal : (ch x times)
    times : (times,) [ms]
    new_fs : [hz]
    '''

    n_samples = int((times[-1] - times[0]) / 1000 * new_fs)
    new_signal_evk, new_times = scipy.signal.resample(signal, n_samples, t=times, axis=1)
    return new_signal_evk, new_times


def baseline_correct(Y, times, delta=0):
    ''' Baseline correct signal using times < delta '''
    newY = np.zeros(Y.shape)
    onset_ix = get_time_index(times, delta)
    baseline_mean = np.mean(Y[:, :onset_ix], axis=1)[np.newaxis]  # shape: (1,dls)
    newY = Y - baseline_mean.T
    assert np.all(np.isclose(np.mean(newY[:, :onset_ix], axis=1), 0, atol=1e-08)), "Baseline mean is not zero"
    return newY


def rereference(Y):
    newY = np.zeros(Y.shape)
    channels_mean = np.mean(Y, axis=0)[np.newaxis]  # shape: (1,dls)
    newY = Y - channels_mean
    assert np.all(np.isclose(np.mean(newY, axis=0), 0, atol=1e-08)), "Baseline mean is not zero"
    return newY


def get_time_index(times, onset=0):
    ''' Returns index of first time greater then delta. For delta=0 gets index of first non-negative time
       OBS: alternative way is to find index of closest value to onset using np.argmin(np.abs(times - onset))
    '''
    return np.sum(times < onset)


