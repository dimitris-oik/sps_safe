import numpy as np
from scipy.stats import ortho_group


# helper functions
def sq_matrix_with_given_cond(n, cond):
    # result is always symmetric positive definite
    log_cond = np.log(cond)
    exp_vec = np.arange(-log_cond/4., log_cond * (n + 1)/(4 * (n - 1)), log_cond/(2.*(n-1)))
    exp_vec = exp_vec[:n]
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)

    return P

def matrix_with_given_cond(n, d, cond, symmetric=False):
    assert d >= 2
    assert n >= d
    P = ortho_group.rvs(dim=n)
    if symmetric:
        Q = P.T
    else:
        Q = ortho_group.rvs(dim=d)
    D = np.zeros((n, d))
    
    t = np.sqrt(cond)
    u = np.random.uniform(low=-1, high=1, size=d-2)
    np.insert(u, 0, -1)
    np.append(u, 1)
    np.fill_diagonal(D, np.float_power(t, u))
    
    A = P@D@Q
    return A






# Non smooth losses
class SVM():
    def __init__(self, n, d, lambd=0.0, A=None, b=None, cond=None):
        self.name = "SVM"
        self.save_name = "svm"
        self.n = n
        self.d = d
        assert self.d >= 2
        assert self.n >= self.d
        self.lambd = lambd
        self.A_cond = cond

        if A is not None:
            self.A = A
        else: # A is None
            if cond is not None and n == d:
                self.A = sq_matrix_with_given_cond(n, cond)
            elif cond is not None and n != d:
                self.A = matrix_with_given_cond(n, d, cond)
            else:
                self.A = np.random.randn(n, d) + 2

        if b is not None:
            self.b = b
        else:
            self.b = np.random.randn(n) + 2

        self.L_i = None
        self.L_max = None
        self.L_avg = None
        self.L = None
        self.mu = None
        self.kappa = None
    
    def func(self, x, ind):
        func_arr = np.array([max(0, 1 - self.b[i] * np.dot(self.A[i], x)) for i in ind])
        return np.average(func_arr) + (self.lambd/2)*np.linalg.norm(x)**2
    
    def grad(self, x, ind):
        grad_arr = -np.array([self.b[i]*self.A[i] if (self.b[i] * np.dot(self.A[i], x)) <= 1 else np.zeros(self.d) for i in ind])
        return np.average(grad_arr, axis=0) + self.lambd*x

class PhaseRetrieval():
    def __init__(self, n, d, consistent):
        self.name = "Phase Retrieval"
        self.save_name = "phase"
        self.n = n
        self.d = d
        assert self.d >= 2
        assert self.n >= self.d

        self.A = np.random.randn(n, d)
        self.x_star = np.random.randn(self.d)
        if consistent:
            self.b = np.dot(self.A, self.x_star)**2
        else:
            self.b = np.random.randn(n)

        self.L_i = None
        self.L_max = None
        self.L_avg = None
        self.L = None
        self.mu = None
        self.kappa = None

    def func(self, x, ind):
        func_arr = np.array([np.abs(np.dot(self.A[i], x)**2-self.b[i]) for i in ind])
        return np.average(func_arr)
    
    def grad(self, x, ind):
        grad_arr = 2*np.array([np.dot(self.A[i], x)*np.sign(np.dot(self.A[i], x)**2-self.b[i])*self.A[i] for i in ind])
        return np.average(grad_arr, axis=0)
