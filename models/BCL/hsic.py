# HSIC regularization and blockwise pairwise-distance utilities.

import torch
import numpy as np
from torch.autograd import Variable, grad


def pairwise_distance_block(a: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """Compute squared pairwise distances in blocks.

    Args:
        a: Feature tensor with shape (D, N).
        block_size: Number of samples processed per block.

    Returns:
        Distance matrix with shape (N, N).
    """
    a = a.t()  # Transpose to shape (N, D).
    n = a.size(0)
    dis = torch.zeros(n, n, device=a.device, dtype=a.dtype)
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        a_block = a[i:i_end]
        a2 = (a_block**2).sum(dim=1, keepdim=True)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            b_block = a[j:j_end]
            b2 = (b_block**2).sum(dim=1, keepdim=True).t()
            ab = a_block @ b_block.t()
            dis[i:i_end, j:j_end] = a2 + b2 - 2 * ab
    return dis


def sigma_estimation(X, Y):
    """sigma from median distance"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    # med =  np.mean(D)
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1e-2:
        med = 1e-2
    return med


def distmat(X):
    """distance matrix"""
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def K_matrix(X, a, block_size: int = 256):
    """
    X: feature matrix, size (fea_dim, sample_size)
    a: vector/matrix for centering H
    """
    # Compute pairwise distances in blocks.
    dis_x = pairwise_distance_block(X, block_size=block_size)  # (N, N)

    sigma = dis_x.mean()  # = 2*sigma^2
    K_x = torch.exp(-dis_x / (sigma + 1e-8))  # Apply the Gaussian kernel.

    # Construct the weighted centering matrix H.
    if a.ndim == 1:  # Handle a one-dimensional weight vector.
        H = torch.diag(a) - torch.mm(a.view(-1, 1), a.view(1, -1))
    else:  # Handle a weight matrix.
        H = a - torch.mm(a, a.t())

    # Kxc = K_x @ H
    Kxc = torch.mm(K_x, H)
    return Kxc


def K_matrix0(X):
    # input.size = (fea_dim*sample_size)
    m = int(X.size()[0])
    dis_x = distmat(X)
    sigma = dis_x.mean()  # = 2*sigma.pow(2)
    K_x = torch.exp(-dis_x / sigma).cuda()
    H = (torch.eye(m) - (1.0 / m) * torch.ones([m, m])).cuda()
    Kxc = torch.mm(K_x, H)
    return Kxc


def kernelmat(X, sigma):
    """kernel matrix baker"""
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])
    Dxx = distmat(X)

    if sigma:
        variance = 2.0 * sigma * sigma * X.size()[1]
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError(
                "Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)
                )
            )
    Kxc = torch.mm(Kx, H)

    return Kxc


def kernelmat0(X, a, sigma):
    """kernel matrix baker"""
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.diag(a) - torch.mm(a, a.T)
    Dxx = distmat(X)

    if sigma:
        variance = 2.0 * sigma * sigma * X.size()[1]
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError(
                "Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)
                )
            )
    Kx = Kx.cuda()
    Kxc = torch.mm(Kx, H)

    return Kxc


def R_operator(K, epsilon):
    n_samples = K.shape[0]
    H = (torch.eye(n_samples) - torch.ones((n_samples, n_samples)) / n_samples).cuda()
    G = torch.mm(torch.mm(H, K), H)
    R = torch.mm(
        G, torch.inverse(G + n_samples * epsilon * torch.eye(n_samples).cuda())
    )
    return R


def c_hsic(X, Y, Z, epsilon=5e-4):
    X_all = torch.cat((X, Z), 1)  # torch.size([d + c, ns + nt])
    Y_all = torch.cat((Y, Z), 1)  # torch.Size([2 + c, ns + nt])

    K_x = kernelmat(X_all, sigma=None).cuda()
    K_y = kernelmat(Y_all, sigma=None).cuda()
    K_z = kernelmat(Z, sigma=None).cuda()

    R_x = R_operator(K_x, epsilon)
    R_y = R_operator(K_y, epsilon)
    R_z = R_operator(K_z, epsilon)

    # R_total = torch.mm(R_y,R_x) -2*torch.mm(torch.mm(R_y,R_x), R_z) + torch.mm(torch.mm(torch.mm(R_y,R_z),R_x),R_z)
    R_total = (
        torch.mm(R_y, R_x)
        - torch.mm(torch.mm(R_y, R_x), R_z)
        - torch.mm(torch.mm(R_y, R_z), R_x)
        + torch.mm(torch.mm(torch.mm(R_y, R_z), R_x), R_z)
    )
    COND_loss = torch.trace(R_total)
    return COND_loss


def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp(-X / (2.0 * sigma * sigma))
    return torch.mean(X)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx = torch.exp(-Dxx / (2.0 * sigma * sigma))  # kernel matrices
        Ky = torch.exp(-Dyy / (2.0 * sigma * sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x, x)
        sy = sigma_estimation(y, y)
        sxy = sigma_estimation(x, y)
        Kx = torch.exp(-Dxx / (2.0 * sx * sx))
        Ky = torch.exp(-Dyy / (2.0 * sy * sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x, y]))
    Dxy = Dxy[: x.size()[0], x.size()[0] :]
    Kxy = torch.exp(-Dxy / (1.0 * sxy * sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2 * torch.mean(Kxy)

    return mmdval


def mmd_pxpy_pxy(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx = torch.exp(-Dxx / (2.0 * sigma * sigma))  # kernel matrices
        Ky = torch.exp(-Dyy / (2.0 * sigma * sigma))
    else:
        sx = sigma_estimation(x, x)
        sy = sigma_estimation(y, y)
        sxy = sigma_estimation(x, y)
        Kx = torch.exp(-Dxx / (2.0 * sx * sx))
        Ky = torch.exp(-Dyy / (2.0 * sy * sy))
    A = torch.mean(Kx * Ky)
    B = torch.mean(torch.mean(Kx, dim=0) * torch.mean(Ky, dim=0))
    C = torch.mean(Kx) * torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2 * B + C
    return mmd_pxpy_pxy_val


def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    Kxc = K_matrix0(x)
    Kyc = K_matrix0(y)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy


def hsic_regular0(x, y, a, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    Kxc = K_matrix(x.T, a)
    Kyc = K_matrix(y.T, a)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy


def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """ """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy / (Px * Py)
    return thehsic


def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True):
    """ """
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma)

    epsilon = 1e-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = Kxc.mm(Kxc_i)
    Ry = Kyc.mm(Kyc_i)
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


def hsic_regular0_block(x, y, a, block_size=256):
    """Accumulate HSIC over blocks to reduce peak memory use.

    Args:
        x, y: Feature tensors with shape (N, D).
        a: Weights with shape (N,) or (N, N).
        block_size: Number of samples processed per block.
    """
    n = x.shape[0]
    Pxy_sum = 0
    count = 0

    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        x_block = x[i:i_end].T  # Transpose the feature block to shape (D, block_size).
        y_block = y[i:i_end].T
        a_block = a[i:i_end] if a.ndim == 1 else a[i:i_end, :]

        Kxc_block = K_matrix(x_block, a_block)
        Kyc_block = K_matrix(y_block, a_block)

        Pxy_sum += torch.sum(Kxc_block * Kyc_block.t())
        count += Kxc_block.numel()

    return Pxy_sum / count
