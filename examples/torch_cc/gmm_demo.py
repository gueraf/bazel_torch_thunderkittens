"""
Example demonstrating how to use the GMM PyTorch wrapper.
"""

import torch

from examples.torch_cc.gmm_torch import gmm_torch_pure


def main():
    """Demonstrate the GMM PyTorch wrapper usage."""
    print("GMM PyTorch Wrapper Demo")
    print("=" * 30)

    # Create example matrices
    K, L, M = 4, 3, 5
    print(f"Matrix dimensions: A({K}x{L}), B({L}x{M}) -> C({K}x{M})")

    # Create random input matrices
    torch.manual_seed(42)  # For reproducible results
    A = torch.randn(K, L, dtype=torch.float32)
    B = torch.randn(L, M, dtype=torch.float32)

    print(f"\nMatrix A:\n{A}")
    print(f"\nMatrix B:\n{B}")

    # Example 1: Basic matrix multiplication (C = A @ B)
    print("\n" + "=" * 50)
    print("Example 1: Basic matrix multiplication (alpha=1.0, beta=0.0)")
    print("=" * 50)

    result1 = gmm_torch_pure(A, B, alpha=1.0, beta=0.0)
    expected1 = torch.mm(A, B)

    print(f"GMM Result:\n{result1}")
    print(f"PyTorch mm Result:\n{expected1}")
    print(f"Results match: {torch.allclose(result1, expected1, rtol=1e-5)}")

    # Example 2: Scaled matrix multiplication (C = 2.0 * A @ B)
    print("\n" + "=" * 50)
    print("Example 2: Scaled matrix multiplication (alpha=2.0, beta=0.0)")
    print("=" * 50)

    result2 = gmm_torch_pure(A, B, alpha=2.0, beta=0.0)
    expected2 = 2.0 * torch.mm(A, B)

    print(f"GMM Result (first few elements):\n{result2[:2, :3]}")
    print(f"Expected (first few elements):\n{expected2[:2, :3]}")
    print(f"Results match: {torch.allclose(result2, expected2, rtol=1e-5)}")

    # Example 3: General matrix multiply with existing output (C = alpha * A @ B + beta * C)
    print("\n" + "=" * 50)
    print("Example 3: GMM with existing output (alpha=1.5, beta=0.5)")
    print("=" * 50)

    C_initial = torch.ones(K, M, dtype=torch.float32)
    alpha, beta = 1.5, 0.5

    result3 = gmm_torch_pure(A, B, alpha, beta, C_initial)
    expected3 = alpha * torch.mm(A, B) + beta * C_initial

    print(f"Initial C:\n{C_initial}")
    print(f"GMM Result (first few elements):\n{result3[:2, :3]}")
    print(f"Expected (first few elements):\n{expected3[:2, :3]}")
    print(f"Results match: {torch.allclose(result3, expected3, rtol=1e-5)}")

    # Example 4: Performance comparison for larger matrices
    print("\n" + "=" * 50)
    print("Example 4: Performance comparison (larger matrices)")
    print("=" * 50)

    # Create larger matrices
    K_large, L_large, M_large = 100, 50, 80
    A_large = torch.randn(K_large, L_large, dtype=torch.float32)
    B_large = torch.randn(L_large, M_large, dtype=torch.float32)

    import time

    # Time the GMM function
    start_time = time.time()
    for _ in range(10):
        result_gmm = gmm_torch_pure(A_large, B_large)
    gmm_time = time.time() - start_time

    # Time PyTorch's built-in mm
    start_time = time.time()
    for _ in range(10):
        result_torch = torch.mm(A_large, B_large)
    torch_time = time.time() - start_time

    print(f"Matrix dimensions: {K_large}x{L_large} @ {L_large}x{M_large}")
    print(f"GMM wrapper time (10 iterations): {gmm_time:.4f}s")
    print(f"PyTorch mm time (10 iterations): {torch_time:.4f}s")
    print(f"Results match: {torch.allclose(result_gmm, result_torch, rtol=1e-5)}")

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
