import numpy as np
import scipy as sp
def aaa(F, Z, tol=1e-13, mmax=100):

    if ~ (type(F) == np.array):
        F = F(Z)
    M = len(Z)
    J = list(range(0, M))
    z = np.empty(0)
    f = np.empty(0)
    C = []
    errors = []
    R = np.mean(F) * np.ones_like(F)
    for m in range(mmax):
        # find largest residual
        j = np.argmax(abs(F - R))
        z = np.append(z, Z[j])
        f = np.append(f, F[j])
        try:
            J.remove(j)
        except:
            pass

        # Cauchy matrix containing the basis functions as columns
        C = 1.0 / (Z[J, None] - z[None, :])
        # Loewner matrix
        A = (F[J, None] - f[None, :]) * C

        # compute weights as right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(A)
        wj = Vh[-1, :].conj()

        # approximation: numerator / denominator
        N = C.dot(wj * f)
        D = C.dot(wj)

        # update residual
        R = F.copy()
        R[J] = N / D

        # check for convergence
        errors.append(np.linalg.norm(F - R, np.inf))
        if errors[-1] <= tol * np.linalg.norm(F, np.inf):
            break

    def r(x): return approximated_function(x, z, f, wj)
    # return z,f,wj
    pol, res, zer = prz(z, f, wj)
    return r, pol, res, zer


def approximated_function(zz, z, f, w, need=False):
    # evaluate r at zz
    zv = np.ravel(zz)  # vectorize zz

    # Cauchy matrix
    CC = 1 / (np.subtract.outer(zv, z))

    # AAA approximation as vector
    r = np.dot(CC, w * f) / np.dot(CC, w)
    if need is True:
        return np.dot(CC, w * f), np.dot(CC, w * f)
    # Find values NaN = Inf/Inf if any
    ii = np.isnan(r)

    # Force interpolation at NaN points
    for j in np.where(ii)[0]:
        r[j] = f[np.where(zv[j] == z)[0][0]]

    # Reshape the result to match the shape of zz
    r = r.reshape(zz.shape)
    return r


def prz(z, f, w):
    m = len(w)
    B = np.eye(m+1)
    B[0, 0] = 0
    E = np.block([[0, w], [np.ones((m, 1)), np.diag(z)]])
    eigvals = sp.linalg.eig(E, B)[0]
    # eigvals[~np.isinf(eigvals)] #remove singularities
    pol = np.real_if_close(eigvals[np.isfinite(eigvals)])
    # Get residues from quotients, in the paper they use a little shift
    # but I coudn't broadcast it correctly
    C = 1.0/(pol[:, None]-z[None, :])
    N = C.dot(f*w)
    # Derivative, formula for simple poles see Zill complex analysis
    D = (-C**2).dot(w)
    res = N/D
    ez = np.block([[0, w], [f[:, None], np.diag(z)]])
    eigvals_zeros = sp.linalg.eig(ez, B)[0]
    zer = eigvals_zeros[~np.isinf(eigvals_zeros)]
    return pol, res, zer


def filter_poles(pol, res):
    pols = []
    ress = []
    for i in range(len(pol)):
        if (np.imag(pol[i]) < 0):
            pols.append(pol[i])
            ress.append(res[i])
    return np.array(pols), np.array(ress)
