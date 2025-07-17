"""Test for the GMM PyTorch wrapper."""

import numpy as np
import torch
from absl.testing import absltest

from examples.torch_cc.gmm_torch import gmm_torch_pure


class GMMTorchTest(absltest.TestCase):

    def test_gmm_basic(self):
        """Test basic GMM functionality with pure PyTorch implementation."""
        # Create test matrices
        K, L, M = 3, 4, 5
        A = torch.randn(K, L, dtype=torch.float32)
        B = torch.randn(L, M, dtype=torch.float32)
        alpha, beta = 2.0, 0.5

        # Test with zero C matrix
        result1 = gmm_torch_pure(A, B, alpha, beta)
        expected1 = alpha * torch.mm(A, B)

        self.assertTrue(torch.allclose(result1, expected1, rtol=1e-5))

        # Test with non-zero C matrix
        C = torch.randn(K, M, dtype=torch.float32)
        result2 = gmm_torch_pure(A, B, alpha, beta, C)
        expected2 = alpha * torch.mm(A, B) + beta * C

        self.assertTrue(torch.allclose(result2, expected2, rtol=1e-5))

    def test_gmm_identity(self):
        """Test GMM with identity matrices."""
        # Test with identity matrix
        size = 4
        A = torch.eye(size, dtype=torch.float32)
        B = torch.randn(size, size, dtype=torch.float32)

        result = gmm_torch_pure(A, B, alpha=1.0, beta=0.0)

        # A @ B where A is identity should equal B
        self.assertTrue(torch.allclose(result, B, rtol=1e-5))

    def test_gmm_dimensions(self):
        """Test that dimension checking works correctly."""
        A = torch.randn(3, 4, dtype=torch.float32)
        B = torch.randn(5, 6, dtype=torch.float32)  # Wrong dimensions

        with self.assertRaises(ValueError):
            gmm_torch_pure(A, B)

    def test_gmm_alpha_beta_coefficients(self):
        """Test that alpha and beta coefficients work correctly."""
        K, L, M = 2, 3, 2
        A = torch.ones(K, L, dtype=torch.float32)
        B = torch.ones(L, M, dtype=torch.float32)
        C = torch.ones(K, M, dtype=torch.float32)

        # With A and B all ones, A @ B should be L * ones
        # So alpha * A @ B + beta * C = alpha * L + beta
        alpha, beta = 2.0, 3.0
        result = gmm_torch_pure(A, B, alpha, beta, C)
        expected = torch.full((K, M), alpha * L + beta, dtype=torch.float32)

        self.assertTrue(torch.allclose(result, expected, rtol=1e-5))

    def test_gmm_zero_matrices(self):
        """Test GMM with zero matrices."""
        K, L, M = 2, 3, 4
        A = torch.zeros(K, L, dtype=torch.float32)
        B = torch.randn(L, M, dtype=torch.float32)
        C = torch.randn(K, M, dtype=torch.float32)

        alpha, beta = 2.0, 3.0
        result = gmm_torch_pure(A, B, alpha, beta, C)

        # Since A is zero, result should be beta * C
        expected = beta * C
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5))

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
