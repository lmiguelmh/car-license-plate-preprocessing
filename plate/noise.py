"""
@author lmiguelmh
@since 20170427
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack



def draw_histogram(img, channels=('r', 'g', 'b'), figsize=(8, 8), mask=None):
    for i, col in enumerate(channels):
        histr = cv2.calcHist([img], [i], mask, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def cv2_fft_magnitude(gray):
    # comentarios por @isipiran - retorna salida compleja a diferencia de np
    # docs: http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#idft

    # se calcula la transformada de Fourier
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    # se intercambian los cuadrantes de la transformada
    dft_shift = np.fft.fftshift(dft)
    # se calcula la magnitud de la transformada
    # se le aplica la función logarítmica, porque las magnitudes
    # suelen ser valores muy altos y queremos reducir
    # a una escala más acotada.
    # se le suma uno para que el logaritmo esté bien definido.
    return np.log(1 + cv2.magnitude(dft_shift[:, :, 0],
                                    dft_shift[:, :, 1]))
    # opencv tutorial multiplies by 20
    # 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))


def np_fft_magnitude(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    return np.log(1 + np.abs(fshift))  # 20*np.log(np.abs(fshift))
    # print(magnitude_spectrum.shape)


def cv2_ifft(gray, mask):
    if mask.shape != gray.shape:
        raise Exception("mask must be the same size of image")

    h, w = gray.shape

    # fourier using opencv
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # convert to mask 3d only when using opencv
    mask[mask == 255] = 1
    mask3d = np.zeros((h, w, 2), np.float32)
    mask3d[:, :, 0] = mask[:, :]
    mask3d[:, :, 1] = mask[:, :]

    # "mask" the fourier
    dft_shift_masked = dft_shift * mask3d
    magnitude_spectrum_masked = np.log(1 + cv2.magnitude(dft_shift_masked[:, :, 0], dft_shift_masked[:, :, 1]))
    # inverse & reconstruct
    f_ishift = np.fft.ifftshift(dft_shift_masked)
    img_back = cv2.idft(f_ishift)

    # img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # http://stackoverflow.com/a/30132485/2692914
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # return magnitude_spectrum_masked, img_back
    return img_back


def homomorphic(gray, low_gamma=0.3, high_gamma=1.5):
    img = gray
    rows, cols = gray.shape

    # based on this excellent answer: http://stackoverflow.com/a/24732434/2692914
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # divides la imagen en alta frecuencia y baja frecuencia
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # las unes con diferentes coeficientes
    gamma1 = low_gamma
    gamma2 = high_gamma
    Ioutlow_gamma = gamma1 * Ioutlow[0:rows, 0:cols]
    Iouthigh_gamma = gamma2 * Iouthigh[0:rows, 0:cols]
    Iout = Ioutlow_gamma + Iouthigh_gamma

    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    return np.array(255 * Ihmf, dtype="uint8")


def draw_comparison(imgs, titles=None, figsize=(12, 12), cmaps=['gray', 'gray'], interpolations=['none', 'none']):
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    if titles:
        a.set_title(titles[0])
    plt.xticks([]), plt.yticks([])
    plt.imshow(imgs[0], cmap=cmaps[0], interpolation=interpolations[0])

    a = fig.add_subplot(1, 2, 2)
    if titles:
        a.set_title(titles[1])
    plt.xticks([]), plt.yticks([])
    plt.imshow(imgs[1], cmap=cmaps[1], interpolation=interpolations[1])
    plt.show()


def fft_filter(gray, mask):
    h, w = gray.shape

    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # magnitude_spectrum = np.log(1 + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    mask3d = np.zeros((h, w, 2), np.float32)
    mask3d[:, :, 0] = mask[:, :]
    mask3d[:, :, 1] = mask[:, :]
    dft_shift_masked = dft_shift * mask3d
    # magnitude_spectrum_masked = np.log(1 + cv2.magnitude(dft_shift_masked[:, :, 0], dft_shift_masked[:, :, 1]))

    f_ishift = np.fft.ifftshift(dft_shift_masked)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
