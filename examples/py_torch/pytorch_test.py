import torch
from absl.testing import absltest


class PytorchTest(absltest.TestCase):
    def test_vector_add(self):
        # Create two tensors
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        # Add them together
        result = a + b

        # Check that the result is correct
        expected = torch.tensor([5, 7, 9])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    absltest.main()
