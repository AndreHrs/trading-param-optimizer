import numpy as np

def sma_filter(N):
    return np.ones(N)/N

def lma_filter(N):
    k = np.arange(N)
    kernel = (2/(N+1)) * (1-k/N)
    return kernel

def ema_filter(N, alpha, normalize=True):
    k = np.arange(N)
    kernel = alpha * (1-alpha) ** k

    if normalize:
        kernel = kernel / np.sum(kernel)
    return kernel

def pad(P,N):
    padding = -np.flip(P[1:N])
    return np.append(padding, P)

def wma(P,N,kernel):
    return np.convolve(pad(P,N), kernel, "valid")