# # Power method for stability analysis
# import numpy as np
# M = np.array([[2., -1., 0.], [-1., 2, -1.], [0., -1., 2.]])  # 3x3 version of 2nd order FTCS
# x = np.array([[1.], [1.], [1.]])
# for i in range(10):
#     x = np.dot(M, x)
#     x /= np.linalg.norm(x)
# #v = x/np.linalg.norm(x)
# v = np.copy(x)
# print(f'Eigenvector: \n{v}')
# mv = np.dot(M, v)
# print(f'Eigenvalue = {mv[0]/v[0]}')
# Mv = M@v
# print(Mv)
# print(f'Eigenvalue = {np.dot(Mv.T, v)/np.dot(v.T, v)}')

def power_iteration(A, n_sims: int):
    b_k = np.random.rand(A.shape[1])
    for _ in range(n_sims):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1/b_k1_norm

    Ab = np.dot(A, b_k)
    lam = np.dot(Ab.T, b_k)/np.dot(b_k.T, b_k)

    return b_k, lam

if __name__ == '__main__':
    import numpy as np
    M = np.array([[2., -1., 0.], [-1., 2, -1.], [0., -1., 2.]])
    eigvec, eigval = power_iteration(M, 20)
    print(f'Eigenvector = {eigvec}\n')
    print(f'Eigenvalue = {eigval}')
