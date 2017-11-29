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
                "--s_test=1024 " \
                "--s_report_interval=40 " \
                "--n_epochs=20 " \
                "--dataset=iscx " \
                % (model_name, model_type, s_batch, n_steps,
                   v_regularization)


def mainold(dice, total):
  model_type_opts = ["flowattmodel"]
  s_batch_opts = [128, 64]
  n_steps_opts = [16]
  v_regularization_opts = [0.001 * i for i in range(5, 50)]

  for i, command_ in enumerate(gen_commands(
    model_type_opts, s_batch_opts, n_steps_opts, v_regularization_opts,
    dice, total)):
    print(command_)

def main(dice, total):
  i = 0
  for batch, reg, name in [(128, 0.032, "b:128,t:flowattmodel,s:16,r:0.032"), (128, 0.0403, "b:128,t:flowattmodel,s:16,r:0.0403"), (128, 0.0413, "b:128,t:flowattmodel,s:16,r:0.0413"), (128, 0.0383, "b:128,t:flowattmodel,s:16,r:0.0383"), (128, 0.038, "b:128,t:flowattmodel,s:16,r:0.038"), (128, 0.018, "b:128,t:flowattmodel,s:16,r:0.018"), (128, 0.018000000000000002, "b:128,t:flowattmodel,s:16,r:0.018000000000000002"), (64, 0.023, "b:64,t:flowattmodel,s:16,r:0.023"), (128, 0.1, "b:128,t:flowattmodel,s:16,r:0.1"), (64, 0.005, "b:64,t:flowattmodel,s:16,r:0.005")]:
    i += 1
    if i % total != dice:
      continue

    print("python3 -m NetDetect.src.main.train " \
          '--model_name="%s" ' \
          "--model_type=flowattmodel " \
          "--s_batch=%s " \
          "--n_steps=16 " \
          "--v_regularization=%s " \
          "--s_test=4096 " \
          "--s_report_interval=40 " \
          "--n_epochs=30 " \
          "--dataset=iscx " \
          % (name, batch, reg))



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-total", "--total", help="Total workers",
                      type=int, required=True)
  parser.add_argument("-dice", "--dice", help="Worker ID",
                      type=int, required=True)
  args = parser.parse_args()

  main(args.dice, args.total)

