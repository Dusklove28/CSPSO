try:
    from absl import flags
except ImportError:
    FLAGS = None
else:
    FLAGS = flags.FLAGS
    FLAGS(['train_sc.py'])


