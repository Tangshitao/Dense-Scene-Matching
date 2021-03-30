import torch
import os.path as osp
from .tester import Tester
from .trainer import Trainer
from utils.logger import Logger
from model.arch.DSMNet import dsm_net
from model.basic import gen_dummy_input, gen_dummy_head_input
from model.head import GeneralHead


class Launcher(object):
    def __init__(self, cfg, test_only=False):
        self.cfg = cfg
        self.model = dsm_net(cfg.MODEL).cuda()

        logger = Logger(cfg.LOG, None, None) if "LOG" in cfg else None

        self.trainer = None
        if not test_only:
            self.trainer = Trainer(cfg, self.model, logger)
        self.tester = Tester(cfg, self.model, logger)

        if cfg.TRAIN.auto_resume:
            self.resume_model(cfg.LOG.path)

    def run_train(self):
        for i in range(self.cfg.TRAIN.train_iters):
            self.trainer.train_iters(self.cfg.TRAIN.model_save_iters)
            self.save_model(self.cfg.LOG.path)

    def save_model(self, log_dir):
        save_params = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "train_niter": self.trainer.niter,
            "test_niter": self.tester.niter,
            "reproj_loss_start": self.trainer.reproj_loss_start,
        }
        checkpoint_name = "checkpoint.pth-{}".format(self.trainer.niter)
        with open(osp.join(log_dir, "latest.txt"), "w") as f:
            f.write(checkpoint_name)
        torch.save(save_params, osp.join(log_dir, checkpoint_name))

    def resume_model(self, log_dir):
        path = osp.join(log_dir, "latest.txt")
        if not osp.exists(path):
            return
        with open(path, "r") as f:
            model_path = f.readline().strip("\n")
        params = torch.load(osp.join(log_dir, model_path))
        optim_dict = params["optimizer"]
        self.model.load_state_dict(params["state_dict"], strict=True)
        self.tester.niter = params["test_niter"] + 1

        if self.trainer is not None:
            if not self.cfg.TRAIN.reset_optimizer:
                self.trainer.optimizer.load_state_dict(optim_dict)
            self.trainer.niter = params["train_niter"]
            self.trainer.reproj_loss_start = params["reproj_loss_start"]

    def run_test(self):
        results = self.tester.run(
            self.cfg.TEST.max_iters,
            self.cfg.TEST.eval_pose,
            True,
            self.cfg.TEST.results_dir,
        )
        print(results)
        return results
