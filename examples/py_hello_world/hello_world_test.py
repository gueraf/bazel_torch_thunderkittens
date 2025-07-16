from absl.testing import absltest

from examples.py_hello_world.hello_world_lib import make_hello_message


class HelloWorldLibTest(absltest.TestCase):
    def test_make_hello_message(self):
        self.assertEqual(make_hello_message("Fabian"), "Hello, Fabian!")


if __name__ == "__main__":
    absltest.main()
