"""
PyTorch wrapper for the GMM function using a shared library approach.
"""

import ctypes
import os
from typing import Optional

import numpy as np
import torch


class GMMWrapper:
    """Wrapper class for the GMM function that integrates with PyTorch."""

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the GMM wrapper.

        Args:
            lib_path: Path to the shared library containing the GMM function.
                     If None, will try to find it automatically.
        """
        if lib_path is None:
            # Try to find the library automatically
            lib_path = self._find_library()

        # Load the shared library
        self.lib = ctypes.CDLL(lib_path)

        # Define the function signature
        # void gmm(const float* A, const float* B, float alpha, float beta,
        #          int K, int L, int M, float* C)
        self.lib.gmm_c_wrapper.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.c_float,  # alpha
            ctypes.c_float,  # beta
            ctypes.c_int,  # K
            ctypes.c_int,  # L
            ctypes.c_int,  # M
            ctypes.POINTER(ctypes.c_float),  # C
        ]
        self.lib.gmm_c_wrapper.restype = None

    def _find_library(self) -> str:
        """Find the shared library automatically."""
        # This would need to be implemented based on your build system
        # For now, assume it's in a standard location
        possible_paths = [
            "./libgmm_wrapper.so",
            "../thunder_kittens/libgmm_wrapper.so",
            "/usr/local/lib/libgmm_wrapper.so",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError("Could not find GMM shared library")

    def gmm_torch(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        alpha: float,
        beta: float,
        C: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform General Matrix Multiply: C = alpha * A @ B + beta * C

        Args:
            A: Input matrix of shape (K, L)
            B: Input matrix of shape (L, M)
            alpha: Scalar multiplier for A @ B
            beta: Scalar multiplier for C
            C: Optional output matrix of shape (K, M). If None, creates zeros.

        Returns:
            Result tensor of shape (K, M)
        """
        # Validate inputs
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("A and B must be 2-dimensional tensors")

        K, L = A.shape
        L_B, M = B.shape

        if L != L_B:
            raise ValueError(f"Matrix dimensions don't match: A is {K}x{L}, B is {L_B}x{M}")

        # Ensure tensors are float32 and contiguous on CPU
        A_cpu = A.cpu().contiguous().float()
        B_cpu = B.cpu().contiguous().float()

        if C is None:
            C_cpu = torch.zeros(K, M, dtype=torch.float32)
        else:
            if C.shape != (K, M):
                raise ValueError(f"C must have shape ({K}, {M}), got {C.shape}")
            C_cpu = C.cpu().contiguous().float()

        # Get data pointers
        A_ptr = A_cpu.data_ptr()
        B_ptr = B_cpu.data_ptr()
        C_ptr = C_cpu.data_ptr()

        # Convert to ctypes pointers
        A_c = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float))
        B_c = ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_float))
        C_c = ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_float))

        # Call the C function
        self.lib.gmm_c_wrapper(A_c, B_c, alpha, beta, K, L, M, C_c)

        return C_cpu


# Global instance for easy access
_gmm_wrapper = None


def get_gmm_wrapper() -> GMMWrapper:
    """Get the global GMM wrapper instance."""
    global _gmm_wrapper
    if _gmm_wrapper is None:
        _gmm_wrapper = GMMWrapper()
    return _gmm_wrapper


def gmm_torch(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch-friendly GMM function.

    Args:
        A: Input matrix of shape (K, L)
        B: Input matrix of shape (L, M)
        alpha: Scalar multiplier for A @ B (default: 1.0)
        beta: Scalar multiplier for C (default: 0.0)
        C: Optional output matrix of shape (K, M). If None, creates zeros.

    Returns:
        Result tensor of shape (K, M)
    """
    wrapper = get_gmm_wrapper()
    return wrapper.gmm_torch(A, B, alpha, beta, C)


# Alternative pure PyTorch implementation for testing/fallback
def gmm_torch_pure(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of GMM for testing/fallback.

    Args:
        A: Input matrix of shape (K, L)
        B: Input matrix of shape (L, M)
        alpha: Scalar multiplier for A @ B (default: 1.0)
        beta: Scalar multiplier for C (default: 0.0)
        C: Optional output matrix of shape (K, M). If None, creates zeros.

    Returns:
        Result tensor of shape (K, M)
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A and B must be 2-dimensional tensors")

    K, L = A.shape
    L_B, M = B.shape

    if L != L_B:
        raise ValueError(f"Matrix dimensions don't match: A is {K}x{L}, B is {L_B}x{M}")

    # Ensure both A and B have the same dtype by promoting to the higher precision type
    if A.dtype != B.dtype:
        # Promote to the higher precision dtype
        if A.dtype == torch.double or B.dtype == torch.double:
            target_dtype = torch.double
        elif A.dtype == torch.float or B.dtype == torch.float:
            target_dtype = torch.float
        else:
            target_dtype = torch.float  # Default fallback

        A = A.to(target_dtype)
        B = B.to(target_dtype)

    if C is None:
        C = torch.zeros(K, M, dtype=A.dtype, device=A.device)
    else:
        if C.shape != (K, M):
            raise ValueError(f"C must have shape ({K}, {M}), got {C.shape}")
        # Ensure C has the same dtype as A and B
        C = C.to(A.dtype)

    # Perform the operation: C = alpha * A @ B + beta * C
    result = alpha * torch.mm(A, B) + beta * C
    return result
