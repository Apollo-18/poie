import numpy as np
from tqdm import tqdm_notebook as tqdm
import scipy as sp
import numba

def to_uint8(img):
    """ Convert to uint8 and clip"""
    return np.clip(img, 0, 255).astype(np.uint8)

def laplacian(img):
    """ Laplacian """
    return (np.roll(img, 1, 0) + np.roll(img, -1, 0) +
            np.roll(img, 1, 1) + np.roll(img, -1, 1) -
            4 * img)

def laplacian_absmax(img1, img2):
    """Max abs laplacian (get max for each term separately)"""
    def absmax(a, b):
        return np.where(np.abs(a) > np.abs(b), a, b)
    
    res = np.zeros_like(img1)
    for axis in [0, 1]:
        for delta in [-1, 1]:
            res += absmax(np.roll(img1, delta, axis) - img1,
                          np.roll(img2, delta, axis) - img2)
    return res

@numba.jit
def poisson1(ix_i, ix_j, sol, rhs):
    """ One step of Gauss - Seidel method
    Returns maximum change for iteration """
    change = 0
    for k in range(len(ix_i)):
        i = ix_i[k]
        j = ix_j[k]
        for c in range(3):
            new_value = (sol[i - 1, j, c] + sol[i + 1, j, c] +
                         sol[i, j - 1, c] + sol[i, j + 1, c] -
                         rhs[i, j, c]) / 4
            change = max(np.abs(new_value - sol[i, j, c]), change)
            sol[i, j, c] = new_value
    return change

def poisson(n, mask, sol, rhs):
    """ Gauss - Seidel: n iterations
    Returns the resulting image and maximum change of sol """
    assert sol.shape[:2] == mask.shape[:2] == rhs.shape[:2], 'Dimensions should be equal'
    nz = mask.nonzero()
    changes_norms = []
    for i in tqdm(range(n)):
        change = poisson1(*nz, sol, rhs)
        changes_norms.append(change)
    return sol, np.array(changes_norms)

def image_cloning(n_iter, mask, back, image):
    return poisson(n_iter, mask, back.copy(), laplacian_absmax(image, back))[0]