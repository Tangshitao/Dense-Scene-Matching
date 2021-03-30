import argparse
import yaml
import os
import random
import numpy as np
import sys

sys.path.append("libs")

from utils.base import *
from engine.launcher import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(10)
random.seed(12)
np.random.seed(2)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/7scenes.yaml")


def main():
    args = parser.parse_args()
    cfg = AttrDict(yaml.load(open(args.config)))
    launcher = Launcher(cfg)
    launcher.run_train()


if __name__ == "__main__":
    main()
