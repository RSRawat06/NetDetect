def gen_commands(model_type_opts, s_batch_opts,
                 n_steps_opts, v_regularization_opts):
  for s_batch in s_batch_opts:
    for model_type in model_type_opts:
      for n_steps in n_steps_opts:
        for v_regularization in v_regularization_opts:
          model_name = "b:%s,t:%s,s:%s,r:%s" % (s_batch, model_type, n_steps,
                                                v_regularization)
          yield "python3 -m NetDetect.src.main_iscx.train " \
                "--model_name=%s " \
                "--model_type=%s " \
                "--s_batch=%s " \
                "--n_steps=%s " \
                "--v_regularization=%s " \
                "--s_test=4096 " \
                "--s_report_interval=2400 " \
                "--n_epochs=40 " \
                % (model_name, model_type, s_batch, n_steps,
                   v_regularization)


def main():
  model_type_opts = ["flowattmodel", "flowmodel"]
  s_batch_opts = [512, 128]
  n_steps_opts = [16, 28]
  v_regularization_opts = [0.1, 0.01, 0.4, 0.8]
  for i, command_ in enumerate(gen_commands(model_type_opts, s_batch_opts,
                               n_steps_opts, v_regularization_opts)):
    print(command_)


if __name__ == "__main__":
  main()

