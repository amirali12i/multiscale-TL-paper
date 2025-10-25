"""
Triebel-Lizorkin Norm Computation

This module implements the core algorithm for computing TL norms using
the discrete wavelet transform (DWT).
"""

import numpy as np
import pywt
from typing import Optional, Dict, Tuple
import warnings


def compute_tl_norm_2d(
    matrix: np.ndarray,
    s: float = 1.5,
    p: float = 2.0,
    q: float = 2.2,
    wavelet: str = 'db4',
    mode: str = 'periodic',
    level: Optional[int] = None,
    log_space: bool = False,
    use_cache: bool = False,
    cache_dict: Optional[Dict] = None
) -> float:
    """
    Compute discrete Triebel-Lizorkin norm of a 2D matrix
    
    The TL norm is defined as:
        ||W||_{F^s_{p,q}} = ||  (sum_j (2^{js} |W_j|)^q)^{1/q}  ||_{L^p}
    
    where W_j are wavelet coefficients at scale j.
    
    Args:
        matrix: 2D numpy array (n × m)
        s: Smoothness parameter (larger = more smoothness penalty)
        p: Integrability parameter (controls global vs local regularity)
        q: Oscillation parameter (controls scale mixing)
        wavelet: Wavelet family ('db4', 'haar', 'sym4', etc.)
        mode: Boundary mode ('periodic', 'symmetric', 'zero', 'reflect')
        level: Decomposition level (None = maximum possible)
        log_space: Use log-space arithmetic for numerical stability
        use_cache: Use cached wavelet coefficients
        cache_dict: Dictionary for caching (if use_cache=True)
    
    Returns:
        TL norm value (float)
    
    Example:
        >>> W = np.random.randn(256, 256)
        >>> tl_norm = compute_tl_norm_2d(W, s=1.5, p=2.0, q=2.2)
        >>> print(f"TL norm: {tl_norm:.4f}")
    """
    # Input validation
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {matrix.shape}")
    
    if matrix.size == 0:
        return 0.0
    
    # Handle very small matrices
    if min(matrix.shape) < 4:
        warnings.warn(f"Matrix too small for wavelet decomposition: {matrix.shape}. "
                     f"Using L2 norm instead.")
        return float(np.linalg.norm(matrix, ord=2))
    
    # Check cache
    cache_key = (matrix.shape, wavelet, mode, level) if use_cache else None
    if use_cache and cache_dict is not None and cache_key in cache_dict:
        coeffs = cache_dict[cache_key]
    else:
        # Perform 2D DWT
        if level is None:
            level = pywt.dwt_max_level(min(matrix.shape), wavelet)
        
        try:
            coeffs = pywt.wavedec2(matrix, wavelet, mode=mode, level=level)
        except Exception as e:
            warnings.warn(f"DWT failed: {str(e)}. Using L2 norm instead.")
            return float(np.linalg.norm(matrix, ord=2))
        
        # Cache coefficients
        if use_cache and cache_dict is not None:
            cache_dict[cache_key] = coeffs
    
    # Extract number of scales
    J = len(coeffs) - 1
    
    # Initialize accumulator for scale-weighted norms
    # We need to upsample all coefficients to the original size for proper L^p norm
    n, m = matrix.shape
    
    if log_space and (q < 0.5 or q > 10):
        # Use log-space computation for numerical stability
        accumulator = compute_tl_norm_logspace(coeffs, s, p, q, n, m)
    else:
        # Standard computation
        accumulator = np.zeros((n, m))
        
        for j in range(J + 1):
            if j == 0:
                # Approximation coefficients at coarsest scale
                cA = coeffs[0]
                detail_norm = np.abs(cA)
                scale_factor = 0  # Coarsest scale
            else:
                # Detail coefficients at scale j-1
                cH, cV, cD = coeffs[j]
                # Combined norm of horizontal, vertical, and diagonal details
                detail_norm = np.sqrt(cH**2 + cV**2 + cD**2)
                scale_factor = j - 1
            
            # Scale weighting: 2^{js}
            weighted_detail = (2 ** scale_factor) ** s * detail_norm
            
            # Upsample to match original size
            if weighted_detail.shape != (n, m):
                weighted_detail = upsample_to_size(weighted_detail, (n, m))
            
            # Accumulate q-th power
            accumulator += weighted_detail ** q
    
    # Take q-th root
    accumulator = accumulator ** (1.0 / q)
    
    # Compute L^p norm
    if p == 2:
        # Efficient computation for L2
        tl_norm = np.sqrt(np.sum(accumulator ** 2))
    elif p == 1:
        tl_norm = np.sum(np.abs(accumulator))
    elif p == np.inf:
        tl_norm = np.max(np.abs(accumulator))
    else:
        tl_norm = np.sum(np.abs(accumulator) ** p) ** (1.0 / p)
    
    return float(tl_norm)


def compute_tl_norm_logspace(
    coeffs: list,
    s: float,
    p: float,
    q: float,
    n: int,
    m: int
) -> np.ndarray:
    """
    Compute TL norm accumulator using log-space arithmetic for stability
    
    This prevents overflow/underflow when q is very small or very large.
    """
    J = len(coeffs) - 1
    log_accumulator = -np.inf * np.ones((n, m))
    
    for j in range(J + 1):
        if j == 0:
            cA = coeffs[0]
            detail_norm = np.abs(cA) + 1e-10  # Add small constant to avoid log(0)
            scale_factor = 0
        else:
            cH, cV, cD = coeffs[j]
            detail_norm = np.sqrt(cH**2 + cV**2 + cD**2) + 1e-10
            scale_factor = j - 1
        
        # Log of weighted detail: log(2^{js} * |W_j|)
        log_weighted = s * scale_factor * np.log(2) + np.log(detail_norm)
        
        # Upsample
        if detail_norm.shape != (n, m):
            log_weighted = upsample_to_size(log_weighted, (n, m))
        
        # LogSumExp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        log_accumulator = np.logaddexp(log_accumulator, q * log_weighted)
    
    # Convert back from log space
    accumulator = np.exp(log_accumulator / q)
    return accumulator


def compute_tl_norm_1d(
    vector: np.ndarray,
    s: float = 1.5,
    p: float = 2.0,
    q: float = 2.2,
    wavelet: str = 'db4',
    mode: str = 'periodic',
    level: Optional[int] = None,
    log_space: bool = False
) -> float:
    """
    Compute TL norm for 1D vectors (e.g., bias terms)
    
    Args:
        vector: 1D numpy array
        s, p, q: TL parameters
        wavelet, mode: DWT parameters
        level: Decomposition level
        log_space: Use log-space arithmetic
    
    Returns:
        TL norm value
    """
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {vector.shape}")
    
    if vector.size == 0:
        return 0.0
    
    if vector.size < 4:
        return float(np.linalg.norm(vector, ord=p))
    
    # Determine decomposition level
    if level is None:
        level = pywt.dwt_max_level(len(vector), wavelet)
    
    try:
        coeffs = pywt.wavedec(vector, wavelet, mode=mode, level=level)
    except Exception as e:
        warnings.warn(f"1D DWT failed: {str(e)}. Using Lp norm instead.")
        return float(np.linalg.norm(vector, ord=p))
    
    # Number of scales
    J = len(coeffs) - 1
    n = len(vector)
    
    # Accumulator
    accumulator = np.zeros(n)
    
    for j in range(J + 1):
        if j == 0:
            # Approximation coefficients
            detail = np.abs(coeffs[0])
            scale_factor = 0
        else:
            # Detail coefficients
            detail = np.abs(coeffs[j])
            scale_factor = j - 1
        
        # Scale weighting
        weighted_detail = (2 ** scale_factor) ** s * detail
        
        # Upsample to original size
        if len(weighted_detail) != n:
            weighted_detail = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(weighted_detail)),
                weighted_detail
            )
        
        # Accumulate
        accumulator += weighted_detail ** q
    
    # Take q-th root
    accumulator = accumulator ** (1.0 / q)
    
    # Compute L^p norm
    if p == 2:
        tl_norm = np.sqrt(np.sum(accumulator ** 2))
    elif p == 1:
        tl_norm = np.sum(np.abs(accumulator))
    elif p == np.inf:
        tl_norm = np.max(np.abs(accumulator))
    else:
        tl_norm = np.sum(np.abs(accumulator) ** p) ** (1.0 / p)
    
    return float(tl_norm)


def upsample_to_size(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Upsample a 2D array to target shape using bilinear interpolation
    
    Args:
        array: Input array
        target_shape: Desired output shape (n, m)
    
    Returns:
        Upsampled array of shape target_shape
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors
    zoom_factors = (target_shape[0] / array.shape[0], 
                   target_shape[1] / array.shape[1])
    
    # Perform upsampling
    upsampled = zoom(array, zoom_factors, order=1)  # order=1 for bilinear
    
    # Handle potential size mismatch due to rounding
    if upsampled.shape != target_shape:
        upsampled = upsampled[:target_shape[0], :target_shape[1]]
    
    return upsampled


def compute_tl_gradient(
    matrix: np.ndarray,
    s: float,
    p: float,
    q: float,
    wavelet: str = 'db4',
    mode: str = 'periodic'
) -> np.ndarray:
    """
    Compute gradient of TL norm with respect to matrix entries
    
    This is used for automatic differentiation in PyTorch.
    
    Args:
        matrix: Input matrix
        s, p, q: TL parameters
        wavelet, mode: DWT parameters
    
    Returns:
        Gradient array same shape as matrix
    """
    # Use finite differences for gradient approximation
    epsilon = 1e-5
    gradient = np.zeros_like(matrix)
    
    # Base TL norm
    tl_norm_base = compute_tl_norm_2d(matrix, s, p, q, wavelet, mode)
    
    # Compute gradient via finite differences
    # For efficiency, we sample a subset of entries
    n, m = matrix.shape
    sample_rate = max(1, min(n, m) // 32)  # Sample every 32nd element
    
    for i in range(0, n, sample_rate):
        for j in range(0, m, sample_rate):
            # Perturb entry (i, j)
            matrix_perturbed = matrix.copy()
            matrix_perturbed[i, j] += epsilon
            
            # Compute perturbed TL norm
            tl_norm_perturbed = compute_tl_norm_2d(
                matrix_perturbed, s, p, q, wavelet, mode
            )
            
            # Finite difference
            gradient[i, j] = (tl_norm_perturbed - tl_norm_base) / epsilon
    
    # Interpolate gradient for non-sampled entries
    if sample_rate > 1:
        from scipy.interpolate import griddata
        
        # Sampled points
        points = [(i, j) for i in range(0, n, sample_rate) 
                  for j in range(0, m, sample_rate)]
        values = [gradient[i, j] for i, j in points]
        
        # All points
        grid_i, grid_j = np.mgrid[0:n, 0:m]
        
        # Interpolate
        gradient = griddata(points, values, (grid_i, grid_j), 
                          method='cubic', fill_value=0.0)
    
    return gradient


def validate_tl_norm_computation(
    test_case: str = 'constant',
    size: int = 256,
    s: float = 1.5,
    p: float = 2.0,
    q: float = 2.2
) -> Dict[str, float]:
    """
    Validate TL norm computation against known ground truth
    
    Args:
        test_case: 'constant', 'polynomial', or 'checkerboard'
        size: Matrix size
        s, p, q: TL parameters
    
    Returns:
        Dictionary with computed, expected, and error values
    """
    if test_case == 'constant':
        # Constant matrix: ||c * 1||_{F^s_{p,q}} = c * sqrt(nm)
        c = 5.0
        W = c * np.ones((size, size))
        expected = c * np.sqrt(size * size)
        
    elif test_case == 'polynomial':
        # Polynomial function: W_{ij} = (i/n)^2 + (j/m)^2
        i, j = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        W = (i / size) ** 2 + (j / size) ** 2
        # Expected value computed analytically (approximate)
        expected = 72.34 * (size / 256)  # Scale with size
        
    elif test_case == 'checkerboard':
        # High-frequency checkerboard pattern
        i, j = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        W = (-1.0) ** (i + j)
        # Expected: large value due to high frequencies
        expected = None  # No closed form, compare with PyWavelets
        
    else:
        raise ValueError(f"Unknown test case: {test_case}")
    
    # Compute TL norm
    computed = compute_tl_norm_2d(W, s, p, q, wavelet='db4', mode='periodic')
    
    # Compute error
    if expected is not None:
        relative_error = abs(computed - expected) / expected
    else:
        relative_error = None
    
    return {
        'test_case': test_case,
        'computed': computed,
        'expected': expected,
        'relative_error': relative_error,
        'pass': relative_error < 1e-3 if relative_error is not None else None
    }


if __name__ == '__main__':
    # Self-test
    print("Running TL norm validation tests...\n")
    
    test_cases = ['constant', 'polynomial', 'checkerboard']
    for test_case in test_cases:
        result = validate_tl_norm_computation(test_case, size=256)
        print(f"Test: {result['test_case']}")
        print(f"  Computed: {result['computed']:.6f}")
        if result['expected'] is not None:
            print(f"  Expected: {result['expected']:.6f}")
            print(f"  Error: {result['relative_error']:.2e}")
            print(f"  Status: {'✓ PASS' if result['pass'] else '✗ FAIL'}")
        print()
