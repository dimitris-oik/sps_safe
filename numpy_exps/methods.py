import numpy as np

def GD(loss, K, x0, gamma=None, return_x=False):
    # x^{k+1} = x^k - \gamma * \nabla f(x^k)
    full_batch = np.arange(loss.n)
    f = np.zeros(K+1)
    f[0] = loss.func(x0, full_batch)
    x = [x0 for i in range(K+1)]

    if gamma is None:
        gamma = 1/loss.L
        name = r'GD, $\gamma=\frac{1}{L}$'
    else:
        name = r'GD, $\gamma={:.3f}'.format(gamma)+'$'

    for k in range(K):
        x[k+1] = x[k] - gamma * loss.grad(x[k], full_batch)
        f[k+1] = loss.func(x[k+1], full_batch)

    if return_x:
        return name, f, x
    else:
        return name, f



def sgd(loss, trials, record_f, x0, gamma, bs):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]
    
    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))

    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)

            x[k+1] = x[k] - gamma * g

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = gamma
                norms[counter, trial] = norm_g
                counter += 1

    name = r'SGD, $\gamma='+"{:.4f}".format(gamma)+r'$'
    return name, f, ss, norms

def sgd_sps_max(loss, trials, record_f, x0, c, g_b, eps, bs):
    # \gamma_k = \min{(f_i(x^k)-f_i^*)/[c*|\nabla f_i(x^k)|^2+eps], \g_b}
    # x^{k+1} = x^k - \gamma_k * \nabla f_i(x^k)
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]
    
    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    tru = np.zeros(trials)
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)
            
            gamma = min(loss.func(x[k], i_k)/(c*norm_g**2+eps), g_b)
            if gamma != g_b:
                tru[trial] += 1
            x[k+1] = x[k] - gamma * g
            
            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = gamma
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'SGD_SPS_max, $c={:.3f}'.format(c)+r'$, $b={:.3f}'.format(g_b)+r'$'
    return name, f, ss, norms, tru

def sgd_sps_plus(loss, trials, record_f, x0, x_star, eps, bs):
    # \gamma_k = [f_i(x^k)-f_i(x^*)]_+/[|\nabla f_i(x^k)|^2+eps]
    # x^{k+1} = x^k - \gamma_k * \nabla f_i(x^k)
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]
    
    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)
            
            gamma = max(loss.func(x[k], i_k)-loss.func(x_star, i_k), 0)/(norm_g**2+eps)
            x[k+1] = x[k] - gamma * g

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = gamma
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'SHD_SPS_+'
    return name, f, ss, norms

def sgd_sps_m(loss, trials, record_f, x0, M, eps, bs):
    # \gamma_k = (f_i(x^k)-f_i^*)/[c*max{|\nabla f_i(x^k)|^2+eps], M}
    # x^{k+1} = x^k - \gamma_k * \nabla f_i(x^k)
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]
    
    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    tru = np.zeros(trials)
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)

            gamma = loss.func(x[k], i_k)/(max(norm_g**2, M)+eps)
            if max(norm_g**2, M) != M:
                tru[trial] += 1
            x[k+1] = x[k] - gamma * g

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = gamma
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'SGD_SPS_M, $M={:.3f}'.format(M)+r'$'
    return name, f, ss, norms, tru



def ima(loss, trials, record_f, x0, eta, lambd, bs):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]

    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        z = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)

            if k==0:
                z[0] = x0 - eta * g
            else:
                z[k] = z[k-1] - eta * g

            x[k+1] = (lambd*x[k]+z[k])/(lambd+1)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = eta
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'IMA, $\lambda='+"{:.3f}".format(lambd)+r'$, $\eta='+"{:.3f}".format(eta)+r'$'
    return name, f, ss, norms

def ima_sps_plus(loss, trials, record_f, x0, lambd, x_star, eps, bs):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]
    
    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        z = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)

            if k==0:
                eta = max(loss.func(x[k], i_k)-loss.func(x_star, i_k), 0)/(norm_g**2+eps)
                z[0] = x0 - eta * g
            else:
                eta = max(0,loss.func(x[k], i_k)-loss.func(x_star, i_k)+np.inner(g, z[k-1]-x[k]))/(norm_g**2+eps)
                z[k] = z[k-1] - eta * g

            x[k+1] = (lambd*x[k]+z[k])/(lambd+1)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = eta
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'IMA_SPS_+, $\lambda='+"{:.3f}".format(lambd)+r'$'
    return name, f, ss, norms

def ima_sps_m(loss, trials, record_f, x0, lambd, M, eps, bs):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]

    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    tru = np.zeros(trials)
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        z = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)

            if k==0:
                eta = loss.func(x[k], i_k)/(max(norm_g**2, M)+eps)
                if max(norm_g**2, M) != M:
                    tru[trial] += 1
                z[0] = x0 - eta * g
            else:
                eta = max(0,loss.func(x[k], i_k)+np.inner(g, z[k-1]-x[k]))/(max(M, norm_g**2)+eps)
                if max(norm_g**2, M) != M:
                    tru[trial] += 1
                z[k] = z[k-1] - eta * g

            x[k+1] = (lambd*x[k]+z[k])/(lambd+1)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = eta
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'IMA_SPS_M, $\lambda='+"{:.3f}".format(lambd)+r'$, $M={:.3f}'.format(M)+r'$'
    return name, f, ss, norms, tru



def ima_last(loss, trials, record_f, x0, eta, bs):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]

    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        z = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)

            if k==0:
                z[0] = x0 - eta * g
            else:
                z[k] = z[k-1] - eta * g

            x[k+1] = ((k+1)*x[k]+z[k])/(k+2)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = eta
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'IMA, $\lambda_t=t$, $\eta='+"{:.3f}".format(eta)+r'$'
    return name, f, ss, norms

def ima_last_sps_plus(loss, trials, record_f, x0, x_star, eps, bs):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]

    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        z = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)

            if k==0:
                eta = max(loss.func(x[k], i_k)-loss.func(x_star, i_k), 0)/(norm_g**2+eps)
                z[0] = x0 - eta * g
            else:
                eta = max(0,loss.func(x[k], i_k)-loss.func(x_star, i_k)+np.inner(g, z[k-1]-x[k]))/(norm_g**2+eps)
                z[k] = z[k-1] - eta * g

            x[k+1] = ((k+1)*x[k]+z[k])/(k+2)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = eta
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'IMA_last_SPS_+'
    return name, f, ss, norms

def ima_last_sps_m(loss, trials, record_f, x0, M, eps, bs):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]

    ss = np.zeros((len(record_f), trials))
    norms = np.zeros((len(record_f), trials))
    tru = np.zeros(trials)
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        z = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            g = loss.grad(x[k], i_k)
            norm_g = np.linalg.norm(g)

            if k==0:
                eta = loss.func(x[k], i_k)/(max(norm_g**2, M)+eps)
                if max(norm_g**2, M) != M:
                    tru[trial] += 1
                z[0] = x0 - eta * g
            else:
                eta = max(0,loss.func(x[k], i_k)+np.inner(g, z[k-1]-x[k]))/(max(M, norm_g**2)+eps)
                if max(norm_g**2, M) != M:
                    tru[trial] += 1
                z[k] = z[k-1] - eta * g

            x[k+1] = ((k+1)*x[k]+z[k])/(k+2)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                ss[counter, trial] = eta
                norms[counter, trial] = norm_g
                counter += 1
    
    name = r'IMA_last_SPS_M, $\lambda_t=t$, $M={:.3f}'.format(M)+r'$'
    return name, f, ss, norms, tru
