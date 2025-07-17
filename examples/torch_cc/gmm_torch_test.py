"""Test for the GMM PyTorch wrapper."""

import numpy as np
import torch
from absl.testing import absltest

from examples.torch_cc.gmm_torch import gmm_torch, gmm_torch_pure


class GMMTorchTest(absltest.TestCase):

    def test_dummy(self):
        """Dummy test to check basic functionality."""
        # Create test matrices
        K, L, M = 3, 4, 5
        A = torch.randn(K, L, dtype=torch.float32)
        B = torch.randn(L, M, dtype=torch.float32)
        alpha, beta = 2.0, 0.5

        # Test with zero C matrix
        result_cpp = gmm_torch(A, B, alpha, beta)
        self.assertIsNotNone(result_cpp)
        self.assertEqual(result_cpp.shape, (K, M))

    def test_gmm_basic(self):
        """Test basic GMM functionality comparing C++ implementation to pure PyTorch."""
        # Create test matrices
        K, L, M = 3, 4, 5
        A = torch.randn(K, L, dtype=torch.float32)
        B = torch.randn(L, M, dtype=torch.float32)
        alpha, beta = 2.0, 0.5

        # Test with zero C matrix
        result_cpp = gmm_torch(A, B, alpha, beta)
        result_pure = gmm_torch_pure(A, B, alpha, beta)

        if result_cpp is not None:
            self.assertTrue(torch.allclose(result_cpp, result_pure, rtol=1e-5))
        else:
            print("C++ implementation not available, skipping comparison")

        # Test with non-zero C matrix
        # Create separate copies to avoid modification issues
        C = torch.randn(K, M, dtype=torch.float32)
        C_for_cpp = C.clone()
        C_for_pure = C.clone()

        result_cpp_c = gmm_torch(A, B, alpha, beta, C_for_cpp)
        result_pure_c = gmm_torch_pure(A, B, alpha, beta, C_for_pure)

        if result_cpp_c is not None:
            self.assertTrue(torch.allclose(result_cpp_c, result_pure_c, rtol=1e-5))
        else:
            print("C++ implementation not available, skipping comparison")

    def test_gmm_identity(self):
        """Test GMM with identity matrices comparing C++ vs pure PyTorch."""
        # Test with identity matrix
        size = 4
        A = torch.eye(size, dtype=torch.float32)
        B = torch.randn(size, size, dtype=torch.float32)

        # Create separate copies to avoid modification issues
        A_for_cpp = A.clone()
        A_for_pure = A.clone()
        B_for_cpp = B.clone()
        B_for_pure = B.clone()

        result_cpp = gmm_torch(A_for_cpp, B_for_cpp, alpha=1.0, beta=0.0)
        result_pure = gmm_torch_pure(A_for_pure, B_for_pure, alpha=1.0, beta=0.0)

        # Both implementations should agree
        if result_cpp is not None:
            self.assertTrue(torch.allclose(result_cpp, result_pure, rtol=1e-5))
        else:
            print("C++ implementation not available, skipping comparison")

    def test_gmm_dimensions(self):
        """Test that dimension checking works correctly in both implementations."""
        A = torch.randn(3, 4, dtype=torch.float32)
        B = torch.randn(5, 6, dtype=torch.float32)  # Wrong dimensions

        # Both implementations should raise ValueError for mismatched dimensions
        with self.assertRaises(ValueError):
            gmm_torch_pure(A, B)

        with self.assertRaises(Exception):  # C++ wrapper might throw different exception
            gmm_torch(A, B)

    def test_gmm_alpha_beta_coefficients(self):
        """Test that alpha and beta coefficients work correctly in both implementations."""
        K, L, M = 2, 3, 2
        A = torch.ones(K, L, dtype=torch.float32)
        B = torch.ones(L, M, dtype=torch.float32)
        C = torch.ones(K, M, dtype=torch.float32)

        # With A and B all ones, A @ B should be L * ones
        # So alpha * A @ B + beta * C = alpha * L + beta
        alpha, beta = 2.0, 3.0

        # Create separate copies to avoid modification issues
        A_for_cpp = A.clone()
        A_for_pure = A.clone()
        B_for_cpp = B.clone()
        B_for_pure = B.clone()
        C_for_cpp = C.clone()
        C_for_pure = C.clone()

        result_cpp = gmm_torch(A_for_cpp, B_for_cpp, alpha, beta, C_for_cpp)
        result_pure = gmm_torch_pure(A_for_pure, B_for_pure, alpha, beta, C_for_pure)

        # Both implementations should agree
        if result_cpp is not None:
            self.assertTrue(torch.allclose(result_cpp, result_pure, rtol=1e-5))
        else:
            print("C++ implementation not available, skipping comparison")

    def test_gmm_zero_matrices(self):
        """Test GMM with zero matrices comparing both implementations."""
        K, L, M = 2, 3, 4
        A = torch.zeros(K, L, dtype=torch.float32)
        B = torch.randn(L, M, dtype=torch.float32)
        C = torch.randn(K, M, dtype=torch.float32)

        alpha, beta = 2.0, 3.0

        # Create separate copies to avoid modification issues
        A_for_cpp = A.clone()
        A_for_pure = A.clone()
        B_for_cpp = B.clone()
        B_for_pure = B.clone()
        C_for_cpp = C.clone()
        C_for_pure = C.clone()

        result_cpp = gmm_torch(A_for_cpp, B_for_cpp, alpha, beta, C_for_cpp)
        result_pure = gmm_torch_pure(A_for_pure, B_for_pure, alpha, beta, C_for_pure)

        # Both implementations should agree
        if result_cpp is not None:
            self.assertTrue(torch.allclose(result_cpp, result_pure, rtol=1e-5))
        else:
            print("C++ implementation not available, skipping comparison")

    def test_different_dtypes(self):
        """Test that the function handles different dtypes correctly."""
        K, L, M = 2, 3, 2
        A = torch.ones(K, L, dtype=torch.float64)  # float64
        B = torch.ones(L, M, dtype=torch.float32)  # float32

        result = gmm_torch_pure(A, B)

        # Should work without error
        self.assertEqual(result.shape, (K, M))
        self.assertIsNotNone(result)


if __name__ == "__main__":
    absltest.main()
