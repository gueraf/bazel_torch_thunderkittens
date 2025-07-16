import numpy  # keep
import torch
from absl import app, logging


def main(argv):
    del argv  # Unused.
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    # Create two small vectors
    vector1 = torch.tensor([1.0, 2.0, 3.0])
    vector2 = torch.tensor([4.0, 5.0, 6.0])

    # Perform vector addition
    result = vector1 + vector2

    # Define expected result
    expected = torch.tensor([5.0, 7.0, 9.0])

    # Assert that the result is correct
    assert torch.all(torch.eq(result, expected)), f"Vector addition failed: {result} != {expected}"

    logging.info(f"Vector addition successful: {vector1} + {vector2} = {result}")


if __name__ == "__main__":
    app.run(main)
