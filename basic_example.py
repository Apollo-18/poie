import numpy as np
import skimage
import skimage.io
import skimage.morphology
import numba


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def laplacian(img):
    return (np.roll(img, 1, 0) + np.roll(img, -1, 0) +
            np.roll(img, 1, 1) + np.roll(img, -1, 1) -
            4 * img)


def laplacian_absmax(img1, img2):
    def absmax(a, b):
        return np.where(np.abs(a) > np.abs(b), a, b)
    
    res = np.zeros_like(img1)
    for axis in [0, 1]:
        for delta in [-1, 1]:
            res += absmax(np.roll(img1, delta, axis) - img1,
                          np.roll(img2, delta, axis) - img2)
    return res


@numba.jit
def poisson1(mask, sol, rhs):
    assert sol.shape[:2] == mask.shape[:2] == rhs.shape[:2], 'Dimensions should be equal'
    for i in range(1, sol.shape[0] - 1):
        for j in range(1, sol.shape[1] - 1):
            if mask[i, j]:
                for c in range(3):
                    sol[i, j, c] = (sol[i - 1, j, c] + sol[i + 1, j, c] +
                                    sol[i, j - 1, c] + sol[i, j + 1, c] -
                                    rhs[i, j, c]) / 4
                    
                    
def poisson(n, mask, sol, rhs):
    for i in range(n):
        poisson1(mask, sol, rhs)
    return sol


back = skimage.io.imread('back.png').astype(float)
fore = skimage.io.imread('fore.png').astype(float)
mask = skimage.morphology.binary_erosion((fore != 0).any(axis=2),
                                         np.ones((3, 3)))

clone = back.copy()
clone[mask] = fore[mask]
skimage.io.imsave('clone.png', to_uint8(clone))

laplace = poisson(100, mask, back.copy(), np.zeros_like(back))
skimage.io.imsave('laplace.png', to_uint8(laplace))

imported = poisson(100, mask, back.copy(), laplacian(fore))
skimage.io.imsave('import.png', to_uint8(imported))

mixed = poisson(100, mask, back.copy(), laplacian_absmax(fore, back))
skimage.io.imsave('mixed.png', to_uint8(mixed))
