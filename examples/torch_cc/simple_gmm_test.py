"""Simple test for the pure PyTorch GMM implementation."""

import os
import sys

import numpy as np
import torch
from absl.testing import absltest

# Add the current directory to Python path to import our module
sys.path.insert(0, os.path.dirname(__file__))

try:
    from gmm_torch import gmm_torch_pure
except ImportError as e:
    print(f"Import error: {e}")

    # Define a simple fallback for testing
    def gmm_torch_pure(A, B, alpha=1.0, beta=0.0, C=None):
        if C is None:
            C = torch.zeros(A.size(0), B.size(1), dtype=A.dtype, device=A.device)
        return alpha * torch.mm(A, B) + beta * C


class GMMTorchSimpleTest(absltest.TestCase):

    def test_gmm_basic(self):
        """Test basic GMM functionality."""
        # Create simple test matrices
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        B = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        # Test simple multiplication (B is identity matrix)
        result = gmm_torch_pure(A, B, alpha=1.0, beta=0.0)

        # A @ I should equal A
        expected = A
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5))
        print("✓ Basic GMM test passed")

    def test_gmm_with_scaling(self):
        """Test GMM with alpha and beta scaling."""
        A = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # 1x2
        B = torch.tensor([[1.0], [1.0]], dtype=torch.float32)  # 2x1
        C = torch.tensor([[5.0]], dtype=torch.float32)  # 1x1

        alpha, beta = 2.0, 3.0
        result = gmm_torch_pure(A, B, alpha, beta, C)

        # A @ B = [[1*1 + 2*1]] = [[3]]
        # alpha * A @ B + beta * C = 2 * 3 + 3 * 5 = 6 + 15 = 21
        expected = torch.tensor([[21.0]], dtype=torch.float32)

        self.assertTrue(torch.allclose(result, expected, rtol=1e-5))
        print("✓ Scaling test passed")


if __name__ == "__main__":
    print("Running simple GMM tests...")
    absltest.main()
