"""
GSDMC Mathematical Utility Functions
Contains gradient computation, similarity calculation, and interpolation functions
"""

import numpy as np
from numba import njit, prange


def compute_gradient_field(field, spacing):
    
    dx, dy, dz = spacing
    
    # np.gradient returns order (dz, dy, dx)
    grad_z, grad_y, grad_x = np.gradient(field, dz, dy, dx)
    
    # Calculate gradient magnitude
    G = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    return G, (grad_x, grad_y, grad_z)


@njit(parallel=True, cache=True)
def compute_similarity_field_numba(field, sigma):
    
    nz, ny, nx = field.shape
    S = np.zeros_like(field)
    sigma_sq = sigma * sigma
    
    # Parallel processing for each layer
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                center_val = field[k, j, i]
                similarity_sum = 0.0
                count = 0
                
                # 3x3x3 neighborhood
                for dk in range(-1, 2):
                    for dj in range(-1, 2):
                        for di in range(-1, 2):
                            nk = k + dk
                            nj = j + dj
                            ni = i + di
                            
                            # Boundary check
                            if 0 <= nk < nz and 0 <= nj < ny and 0 <= ni < nx:
                                neighbor_val = field[nk, nj, ni]
                                diff_sq = (neighbor_val - center_val) * (neighbor_val - center_val)
                                similarity_sum += np.exp(-diff_sq / sigma_sq)
                                count += 1
                
                if count > 0:
                    S[k, j, i] = similarity_sum / count
    
    return S


def compute_similarity_field(field, sigma=0.4):
    
    return compute_similarity_field_numba(field, sigma)


@njit(cache=True)
def hermite_interpolation_numba(p1, p2, v1, v2, grad1, grad2, iso, max_iter=5):
    
    # 1. Calculate edge direction
    edge_vec = p2 - p1
    edge_len = np.sqrt(edge_vec[0]**2 + edge_vec[1]**2 + edge_vec[2]**2)
    
    if edge_len < 1e-10:
        return (p1 + p2) * 0.5  # Degenerate case
    
    edge_dir = edge_vec / edge_len
    
    # 2. Project directional derivative
    d1 = (grad1[0]*edge_dir[0] + grad1[1]*edge_dir[1] + grad1[2]*edge_dir[2]) * edge_len
    d2 = (grad2[0]*edge_dir[0] + grad2[1]*edge_dir[1] + grad2[2]*edge_dir[2]) * edge_len
    
    # 3. Monotonicity constraint
    delta = v2 - v1
    
    if abs(delta) < 1e-10:
        return (p1 + p2) * 0.5
    
    # Check derivative sign
    if delta * d1 < 0:
        d1 = 0.0
    if delta * d2 < 0:
        d2 = 0.0
    
    # Check derivative magnitude
    max_slope = 3.0 * abs(delta)
    if abs(d1) > max_slope:
        d1 = np.sign(d1) * max_slope
    if abs(d2) > max_slope:
        d2 = np.sign(d2) * max_slope
    
    # 4. Hermite polynomial coefficients
    a = 2*v1 - 2*v2 + d1 + d2
    b = -3*v1 + 3*v2 - 2*d1 - d2
    c = d1
    d = v1
    
    # 5. Newton iteration to solve
    t = (iso - v1) / delta
    t = max(0.0, min(1.0, t))
    
    for _ in range(max_iter):
        H_t = a*t*t*t + b*t*t + c*t + d
        H_prime = 3*a*t*t + 2*b*t + c
        
        if abs(H_prime) < 1e-10:
            break
        
        t_new = t - (H_t - iso) / H_prime
        t_new = max(0.0, min(1.0, t_new))
        
        if abs(t_new - t) < 1e-6:
            break
        
        t = t_new
    
    # 6. Calculate final coordinates
    return p1 + t * edge_vec


def hermite_interpolation(p1, p2, v1, v2, grad1, grad2, iso, max_iter=5):
    
    return hermite_interpolation_numba(p1, p2, v1, v2, grad1, grad2, iso, max_iter)


def linear_interpolation(p1, p2, v1, v2, iso):
    
    delta = v2 - v1
    if abs(delta) < 1e-10:
        return (p1 + p2) / 2.0
    
    t = (iso - v1) / delta
    t = np.clip(t, 0.0, 1.0)
    return p1 + t * (p2 - p1)


def midpoint_interpolation(p1, p2):
    
    return (p1 + p2) / 2.0
