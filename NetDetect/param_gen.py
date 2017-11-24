import argparse


def gen_commands(model_type_opts, s_batch_opts,
                 n_steps_opts, v_regularization_opts,
                 dice, total):
  i = 0
  for s_batch in s_batch_opts:
    for model_type in model_type_opts:
      for n_steps in n_steps_opts:
        for v_regularization in v_regularization_opts:
          i += 1
          if i % total != dice:
            continue

          model_name = "b:%s,t:%s,s:%s,r:%s" % (s_batch, model_type, n_steps,
                                                v_regularization)
          yield "python3 -m NetDetect.src.main.train " \
                '--model_name="%s" ' \
                "--model_type=%s " \
                "--s_batch=%s " \
                "--n_steps=%s " \
                "--v_regularization=%s " \
                "--s_test=4096 " \
                "--s_report_interval=2400 " \
                "--n_epochs=25 " \
                "--dataset=iscx " \
                % (model_name, model_type, s_batch, n_steps,
                   v_regularization)


def test():
  commands = []
  for i in range(0, 5):
    model_type_opts = ["flowattmodel", "flowmodel"]
    s_batch_opts = [512, 128, 64]
    n_steps_opts = [16, 28]
    v_regularization_opts = [0.1, 0.05, 0.01, 0.15, 0.4]

    for i, command_ in enumerate(gen_commands(
      model_type_opts, s_batch_opts, n_steps_opts, v_regularization_opts,
      i, 5)):
      commands.append(command_)
  assert(len(set(commands)) == len(commands))
  assert(len(set(commands)) == 2 * 3 * 2 * 5)


def main(dice, total):
  model_type_opts = ["flowattmodel", "flowmodel"]
  s_batch_opts = [512, 128, 64]
  n_steps_opts = [16, 28]
  v_regularization_opts = [0.1, 0.05, 0.01, 0.15, 0.4]

  for i, command_ in enumerate(gen_commands(
    model_type_opts, s_batch_opts, n_steps_opts, v_regularization_opts,
    dice, total)):
    print(command_)


if __name__ == "__main__":
  test()
  parser = argparse.ArgumentParser()
  parser.add_argument("-total", "--total", help="Total workers",
                      type=int, required=True)
  parser.add_argument("-dice", "--dice", help="Worker ID",
                      type=int, required=True)
  args = parser.parse_args()

  main(args.dice, args.total)

