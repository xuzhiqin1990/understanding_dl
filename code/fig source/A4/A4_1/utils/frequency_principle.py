import numpy as np


def my_fft(data, freq_len=40, x_input=np.zeros(10), kk=0, min_f=0, max_f=np.pi/3, isnorm=1):
    """    
    The function returns the Fourier transform of the time series. 
    
    The function is written in Python. 
    
    The function is called `my_fft`. 
    
    The function takes in the following arguments: 
    
    - `data`:
    
    :param data: the data you want to do fft on
    :param freq_len: the number of frequencies to return, defaults to 40 (optional)
    :param x_input: the time series
    :param kk: the frequencies to be used in the Fourier transform. If kk is not provided, then the
    frequencies are chosen to be between min_f and max_f, with freq_len number of frequencies, defaults
    to 0 (optional)
    :param min_f: minimum frequency to consider, defaults to 0 (optional)
    :param max_f: the maximum frequency you want to consider
    :param isnorm: whether to take absolute value of the fft coefficients, defaults to 1 (optional)
    :return: the fft of the data.
    """
    second_diff_input = np.mean(np.diff(np.diff(np.squeeze(x_input))))
    if abs(second_diff_input) < 1e-10:
        datat = np.squeeze(data)
        datat_fft = np.fft.fft(datat)
        freq_len = min(freq_len, len(datat_fft))
        print(freq_len)
        ind2 = range(freq_len)
        fft_coe = datat_fft[ind2]
        if isnorm == 1:
            return_fft = np.absolute(fft_coe)
        else:
            return_fft = fft_coe
    else:
        return_fft = get_ft_multi(
            x_input, data, kk=kk, freq_len=freq_len, min_f=min_f, max_f=max_f, isnorm=isnorm)
    return return_fft


def get_ft_multi(x_input, data, kk=0, freq_len=100, min_f=0, max_f=np.pi/3, isnorm=1):
    """
    > This function takes in a set of input data, and returns the Fourier transform of the data
    
    :param x_input: the input data, a matrix of size (n, d), where n is the number of samples and d is
    the dimension of the input data
    :param data: the data matrix, each row is a data point
    :param kk: the frequencies to be used in the Fourier transform. If kk is not provided, then the
    frequencies are chosen to be between min_f and max_f, with freq_len number of frequencies, defaults
    to 0 (optional)
    :param freq_len: the number of frequencies to use, defaults to 100 (optional)
    :param min_f: minimum frequency, defaults to 0 (optional)
    :param max_f: the maximum frequency to consider
    :param isnorm: whether to take absolute value of the output, defaults to 1 (optional)
    :return: the Fourier transform of the input data.
    """
    n = x_input.shape[1]
    if np.max(abs(kk)) == 0:
        k = np.linspace(min_f, max_f, num=freq_len, endpoint=True)
        kk = np.matmul(np.ones([n, 1]), np.reshape(k, [1, -1]))
    tmp = np.matmul(np.transpose(data), np.exp(-1J * (np.matmul(x_input, kk))))
    if isnorm == 1:
        return_fft = np.absolute(tmp)
    else:
        return_fft = tmp
    return np.squeeze(return_fft)


def SelectPeakIndex(FFT_Data, endpoint=True):
    """
    It finds the indices of the peaks in a 1D array
    
    :param FFT_Data: the data you want to find the peaks of
    :param endpoint: if True, the first and last data points are also considered as peaks, defaults to
    True (optional)
    :return: The index of the peaks in the FFT data.
    """
    D1 = FFT_Data[1:-1]-FFT_Data[0:-2]
    D2 = FFT_Data[1:-1]-FFT_Data[2:]
    D3 = np.logical_and(D1 > 0, D2 > 0)
    tmp = np.where(D3 == True)
    sel_ind = tmp[0]+1
    if endpoint:
        if FFT_Data[0]-FFT_Data[1] > 0:
            sel_ind = np.concatenate([[0], sel_ind])
        if FFT_Data[-1]-FFT_Data[-2] > 0:
            Last_ind = len(FFT_Data)-1
            sel_ind = np.concatenate([sel_ind, [Last_ind]])
    return sel_ind

def get_fft_abs_err(output_list,idx, args, idx_threshold=3):
    """
    It takes the output of the model, the indices of the top 3 frequencies, and the original data, and
    returns the absolute error of the top 3 frequencies of the model output compared to the original
    data
    
    :param output_list: list of predicted outputs
    :param idx: the indices of the top 3(optional) frequencies in the FFT of the target signal
    :param args: the arguments passed to the model
    :param idx_threshold: the number of frequencies to consider, defaults to 3 (optional)
    :return: The absolute error of the FFT of the predicted output and the FFT of the target output.
    """
    y_pred_epoch = np.squeeze(output_list)
    idx1 = idx[:idx_threshold]
    abs_err = np.zeros([len(idx1), len(output_list)])
    y_fft = my_fft(args.train_targets)
    tmp1 = y_fft[idx1]
    for i in range(len(y_pred_epoch)):
        tmp2 = my_fft(y_pred_epoch[i])[idx1]
        abs_err[:, i] = np.abs(tmp1 - tmp2)/(1e-5 + tmp1)
        return abs_err
