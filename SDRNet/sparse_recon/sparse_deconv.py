import cupy as cp
import numpy as np
import warnings
import time
import gc

from matplotlib import pyplot as plt

from .sparse_hessian_recon.sparse_hessian_recon import sparse_hessian
from .iterative_deconv.iterative_deconv import iterative_deconv
from .iterative_deconv.kernel import Gauss
from .utils.background_estimation import background_estimation
from .utils.upsample import spatial_upsample, fourier_upsample
try:
    import cupy as cp
except ImportError:
    cupy = None
xp = np if cp is None else cp
if xp is not cp:
    warnings.warn("could not import cupy... falling back to numpy & cpu.")
def sparse_deconv(img, sigma, sparse_iter = 100, fidelity = 150, sparsity = 10, tcontinuity = 0.5,
                          background = 1, deconv_iter = 7, deconv_type = 1, up_sample = 0):

    """Sparse deconvolution.
   	----------
   	It is an universal post-processing framework for 
   	fluorescence (or intensity-based) image restoration, 
   	including xy (2D), xy-t (2D along t axis), 
   	and xy-z (3D) images. 
   	It is based on the natural priori 
   	knowledge of forward fluorescent 
   	imaging model: sparsity and 
   	continuity along xy-t(z) axes.
   	----------
    Parameters
    ----------
    img : ndarray
       Input image (can be T × X × Y).
    sigma : 1/2/3 element(s) list
       The point spread function size in pixel.
       [x, y, z] dimension
       3D deconv feature is still in progress,
       now is plane-by-plane 2D deconv. 
    sparse_iter:  int, optional
         the iteration of sparse hessian {default: 100}
    fidelity : int, optional
       fidelity {default: 150}
    tcontinuity  : optional
       continuity along z-axial {default: 0.5}
    sparsity :  int, optional
        sparsity {default: 10}
    background:int, optional
        background estimation {default:1}:
        when background is none, 0
        when background is Weak background (High SNR), 1
        when background is Strong background (High SNR), 2
        when background is Weak background (Low SNR), 3
        when background is medium background (Low SNR), 4
        when background is Strong background (Low SNR), 5
        Weak Background（弱背景）：指的是图像中的背景区域，其特征（亮度、纹理、对比度等）较为模糊、不明显或低对比度，通常与前景物体的边缘不清晰。弱背景的区域可能具有类似于噪声的特征，难以从前景中区分开来。
Strong Background（强背景）：指的是图像中的背景区域，其特征明显、对比度高，通常背景中的纹理、颜色或亮度变化较大，可以清楚地区分于前景。
    deconv_iter : int, optional
        the iteration of deconvolution {example:7}
    deconv_type : int, optional
       choose the different type deconvolution:
       0: No deconvolution       
       1: Richardson-Lucy deconvolution
       2: LandWeber deconvolution
    up_sample : int, optional
       choose the different type upsampling (x2) operation:
       0: No upsampling       
       1: Fourier upsampling
       2: Spatial upsampling (should decrease the fidelity & sparsity)

    Returns
    -------
    img_last : ndarray
       The sparse deconvolved image.

    Examples
    --------
    >>> from sparse_recon.sparse_deconv import sparse_deconv
	>>> from skimage import io
    >>> img = io.imread('test.tif')
	>>> img_recon = sparse_deconv(img, [5,5])
    References
    ----------
      [1] Weisong Zhao et al. Sparse deconvolution improves
      the resolution of live-cell super-resolution 
      fluorescence microscopy, Nature Biotechnology (2022),
      https://doi.org/10.1038/s41587-021-01092-2
    """
    start = time.process_time()
    if not sigma:
        print("The PSF's sigma is not given, turning off the iterative deconv...")
        deconv_type = 0

    img = np.array(img, dtype = 'float32')
    scaler = np.max(img)
    img = img / scaler #标准化

    # 把主体部分提取出来
    # bg=2 的效果不错
    if background == 1:
        backgrounds = background_estimation(img / 2.5)
        img = img - backgrounds
    elif background == 2:
        backgrounds = background_estimation(img / 2)
        img = img - backgrounds
    elif background == 3:
        medVal = np.mean(img) / 2.5
        img[img > medVal] = medVal
        backgrounds = background_estimation(img)
        img = img - backgrounds
    elif background== 4:
        medVal = np.mean(img)/2
        img[img> medVal] = medVal
        backgrounds = background_estimation(img)
        img = img - backgrounds
    elif background == 5:
        medVal = np.mean(img)
        img[img > medVal] = medVal
        backgrounds = background_estimation(img)
        img = img - backgrounds


    img = img / (img.max())
    img[img < 0] = 0

    # 处理我们的显微镜图片的时候不需要补零和上采样
    if up_sample == 1:
        img = fourier_upsample(img)
    elif up_sample == 2:
        img = spatial_upsample(img)

    img = img / (img.max())



    # 垃圾回收机制：gc板块
    gc.collect()
    xp.clear_memo()

    img_sparse = sparse_hessian(img, sparse_iter, fidelity, sparsity, tcontinuity)
    # end = time.process_time()
    # print('sparse-hessian time %0.2fs' % (end - start))
    img_sparse = img_sparse / (img_sparse.max())
    if deconv_type == 0:
        img_last = img_sparse
        end = time.process_time()
        # print('deconv time %0.2fs' % (end - start))
        return scaler * img_last
    else:
        # start = time.process_time()
        kernel = Gauss(sigma)
        img_last = iterative_deconv(img_sparse, kernel, deconv_iter, rule = deconv_type)
        end = time.process_time()
        # print('deconv time %0.2fs' % (end - start))
        return scaler * img_last
