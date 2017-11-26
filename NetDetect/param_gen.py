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
                "--s_test=2048 " \
                "--s_report_interval=400 " \
                "--n_epochs=7 " \
                "--dataset=iscx " \
                % (model_name, model_type, s_batch, n_steps,
                   v_regularization)


def main(dice, total):
  model_type_opts = ["flowattmodel"]
  s_batch_opts = [1024]
  n_steps_opts = [16]
  v_regularization_opts = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

  for i, command_ in enumerate(gen_commands(
    model_type_opts, s_batch_opts, n_steps_opts, v_regularization_opts,
    dice, total)):
    print(command_)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-total", "--total", help="Total workers",
                      type=int, required=True)
  parser.add_argument("-dice", "--dice", help="Worker ID",
                      type=int, required=True)
  args = parser.parse_args()

  main(args.dice, args.total)

