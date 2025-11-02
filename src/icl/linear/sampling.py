import torch

# Sampling Utilities

def orthonormal_basis(vectors):
    basis = []
    for v in vectors:
        for b in basis:
            v = v - torch.dot(v, b) * b
        norm = torch.norm(v)
        if norm > 1e-8:
            basis.append(v / norm)
    return torch.stack(basis, dim=1)

def sample_unit_ball(k, n_samples, device=None, is_on_sphere=False):
    # Sample points uniformly from the unit ball in R^k
    z = torch.randn(n_samples, k, device=device)
    z = z / torch.norm(z, dim=1, keepdim=True)
    if is_on_sphere:
        return z  # Return points on the surface of the unit sphere
    r = torch.rand(n_samples, 1, device=device) ** (1.0 / k)
    return z * r

def sample_union_unit_balls_affine_span_with_weights(
        points: torch.Tensor, n_samples: int, r: float = 1.0, 
        orthogonal_offset: float = 0.0, is_on_sphere: bool = False
        ):
    """
    Args:
        points: Tensor of shape (3, d) — three center points.
        n_samples: Number of samples to generate.

    Returns:
        samples: Tensor of shape (n_samples, d)
        weights: Tensor of shape (n_samples, 3), where each row sums to 1
                 and gives affine combination coefficients: x = w1*p1 + w2*p2 + w3*p3
    """
    if points.ndim == 3:
        points = points.squeeze(-1)  # Convert from (3, d, 1) to (3, d)
    assert points.shape[0] == 3, "Input must be of shape (3, d)"
    d = points.shape[1]
    device = points.device

    p1, p2, p3 = points[0], points[1], points[2]

    # Step 1: Get orthonormal basis for affine span
    v1 = p2 - p1
    v2 = p3 - p1
    B = torch.stack([v1, v2], dim=0)         # (2, d)
    Q = orthonormal_basis(B)                 # (d, k)
    k = Q.shape[1]

    # Step 2: Sample local displacements
    indices = torch.randint(0, 3, (n_samples,), device=device)
    chosen_centers = points[indices]                        # (n_samples, d)
    local_samples = r * sample_unit_ball(k, n_samples, device, is_on_sphere)  # (n_samples, k)
    displacements = local_samples @ Q.T                     # (n_samples, d)
    samples = chosen_centers + displacements                # (n_samples, d)

    if orthogonal_offset > 0 and k < d:
        # Build orthogonal direction
        Q_full, _ = torch.linalg.qr(torch.eye(d, device=device))  # full basis
        Q_proj = Q @ Q.T                            # projection matrix onto span
        residual_basis = Q_full - Q_proj @ Q_full                 # orthogonal component
        orth_basis = torch.linalg.qr(residual_basis)[0]          # orthonormalize
        orth_dir = orth_basis[:, 0]                              # take a consistent orthogonal direction

        samples = samples + orthogonal_offset * orth_dir         # (n_samples, d)
        # points = points + orthogonal_offset * orth_dir
    
    samples = torch.cat([points, samples], dim=0)  # Append original points to samples for affine combination # (3+n_samples, d)

    # Step 3: Compute weights
    P = points.T                      # (d, 3)
    # P_aug = torch.cat([P, torch.ones(1, 3, device=device)], dim=0)        # (d+1, 3)
    # samples_aug = torch.cat([samples.T, torch.ones(1, n_samples+3, device=device)], dim=0)  # (d+1, 3+n_samples)
    #weights = torch.linalg.lstsq(P_aug, samples_aug).solution.T          # (3+n_samples, 3)
    weights = torch.linalg.lstsq(P, samples.T).solution.T  

    return samples, weights


def sample_union_unit_balls(
        points: torch.Tensor, n_samples: int, r: float = 1.0, is_on_sphere: bool = False
        ):
    """
    Args:
        points: Tensor of shape (3, d) — three center points.
        n_samples: Number of samples to generate.

    Returns:
        samples: Tensor of shape (n_samples, d)
        weights: Tensor of shape (n_samples, 3), where each row sums to 1
                 and gives affine combination coefficients: x = w1*p1 + w2*p2 + w3*p3
    """
    if points.ndim == 3:
        points = points.squeeze(-1)  # Convert from (3, d, 1) to (3, d)
    assert points.shape[0] == 3, "Input must be of shape (3, d)"
    d = points.shape[1]
    device = points.device

    # Step 2: Sample local displacements
    indices = torch.randint(0, 3, (n_samples,), device=device)
    chosen_centers = points[indices]                        # (n_samples, d)
    local_samples = r * sample_unit_ball(d, n_samples, device, is_on_sphere)  # (n_samples, k)
    samples = chosen_centers + local_samples                # (n_samples, d)
    
    samples = torch.cat([points, samples], dim=0)  # Append original points to samples for affine combination # (3+n_samples, d)

    return samples


def decompose_points(points, centers):
    C = centers.T  # (d,m)
    C_pinv = torch.linalg.pinv(C)
    W = points @ C_pinv.T
    return W

def sample_points_from_balls(centers, r, n_per_ball=100, dtype=torch.float32, generator=None):
    device = centers.device
    M, d = centers.shape
    all_points = []
        
    for c in centers:
        dirs = torch.randn(n_per_ball, d, device=device, dtype=dtype, generator=generator)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        U = torch.rand(n_per_ball, device=device, dtype=dtype, generator=generator)
        radii = r * U.pow(1.0/d)

        pts = c + dirs * radii[:, None]
        all_points.append(pts)

    points = torch.cat(all_points, dim=0)
    points = torch.cat([centers, points], dim=0)
    W = decompose_points(points, centers)

    return points, W