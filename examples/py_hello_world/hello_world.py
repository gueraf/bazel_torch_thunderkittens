from absl import app, flags, logging

from examples.py_hello_world.hello_world_lib import make_hello_message

flags.DEFINE_string("name", "Fabian", "Name to greet.")

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    message = make_hello_message(FLAGS.name)
    logging.info(message)


if __name__ == "__main__":
    app.run(main)
