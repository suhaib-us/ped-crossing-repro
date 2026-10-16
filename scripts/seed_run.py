import os, sys, random, runpy
import numpy as np
import tensorflow as tf

seed = int(os.environ["SEED"])
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

sys.argv = ["train_test.py", "-c", sys.argv[1]]
runpy.run_path("train_test.py", run_name="__main__")
