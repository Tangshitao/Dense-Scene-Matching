import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle as pkl
from skimage.io import imread
import os.path as osp
import cv2
import os

import random
import numpy as np
from utils.image import *
from utils.reader import *
from utils.geometry import *


class VideoDataset(Dataset):
    def __init__(
        self, cfg, transform,
    ):
        self.ref_topk = cfg.ref_topk
        self.tempo_len = cfg.tempo_len
        self.tempo_interval = cfg.tempo_interval
        self.pad_datasets = cfg.pad_datasets
        self.overlap_check = cfg.overlap_check
        self.depth_filter_ratio = cfg.depth_filter_ratio
        self.max_overlap_thresh = cfg.max_overlap_thresh
        self.min_overlap_thresh = cfg.min_overlap_thresh
        self.depth_error_thresh = cfg.depth_error_thresh
        self.pad_image = cfg.pad_image
        self.img_K = None
        self.depth_K = None
        self.enable_hard_example = False
        self.crop_img_func = None
        self.crop_depth_func = None
        self.max_seq_len = 0
        self.max_seq_num_per_scene = 0

        if cfg.check_topk < 0:
            self.check_topk = cfg.ref_topk
        else:
            self.check_topk = cfg.check_topk

        (
            self.meta_info_list,
            self.start_idxs,
            self.end_idxs,
        ) = self.load_and_fuse_datasets(cfg.base_dir, cfg.seq_list_path)

        self.transform = transform

        self.lmdb_data_reader = (
            None
            if cfg.lmdb_data_path is False
            else LMDBModel(cfg.lmdb_data_path, workers=1)
        )

        self.img_H, self.img_W = 480, 640
        self.valid_idx_list = self.reset_valid_list()
        print(len(self.valid_idx_list))

    def get_valid_list(self):
        return sorted(list(self.valid_idx_list))

    def reset_valid_list(self):
        valid_idx_list = []
        # tot=0
        for i, meta_data in enumerate(self.meta_info_list):
            if (
                not self.enable_hard_example
                and "hard_example" in meta_data
                and meta_data["hard_example"]
            ):
                continue
            topk_list = meta_data["topk"]
            new_topk_list = []
            max_overlap = 0
            for j, meta_data_topk in enumerate(topk_list):

                if self.overlap_check:
                    if (
                        meta_data["overlap"][j] > self.min_overlap_thresh
                        and meta_data["depth_error"][j] < self.depth_error_thresh
                    ):
                        new_topk_list.append(meta_data_topk)
                        max_overlap = max(max_overlap, meta_data["overlap"][j])

                    if max_overlap > self.max_overlap_thresh:
                        meta_data["topk"] = new_topk_list
                    else:
                        meta_data["topk"] = []

            if len(meta_data["topk"]) >= self.check_topk:
                valid_idx_list.append(i)

        return valid_idx_list

    def load_and_fuse_datasets(self, base_dir, seq_list_path):
        datasets = []

        meta_info_list = self.load_meta_info_list(os.path.join(base_dir, seq_list_path))

        datasets += self.load_general_datasets(base_dir, meta_info_list)

        if self.pad_datasets:
            datasets = self.fuse_and_pad_datasets(datasets)

        output_datasets = []
        start_idxs = []
        end_idxs = []
        for ds_seq in datasets:
            start_idx = len(output_datasets)
            end_idx = start_idx + len(ds_seq)

            for meta_info in ds_seq:
                start_idxs.append(start_idx)
                end_idxs.append(end_idx)
                output_datasets.append(meta_info)

        return output_datasets, start_idxs, end_idxs

    def load_general_datasets(self, base_dir, seqs):
        if type(seqs) is list:
            new_seq = []
            for frame in seqs:
                frame["base_dir"] = base_dir
                new_seq.append(frame)
            new_seq = sorted(new_seq, key=lambda k: k["file_name"])
            return [new_seq]

        datasets = []
        for sn, seq in seqs.items():
            new_seq = []
            seq = sorted(seq, key=lambda k: k["file_name"])

            for frame in seq:
                frame["base_dir"] = base_dir
                new_seq.append(frame)
            self.max_seq_len = max(self.max_seq_len, len(new_seq))
            datasets.append(new_seq)

        return datasets

    def load_7scenes_datasets(self, dataset_list):
        datasets = []
        for ds_path, base_dir in dataset_list:
            ds = self.load_meta_info_list(os.path.join(base_dir, ds_path))
            new_ds = []
            for meta_info in ds:
                meta_info["base_dir"] = base_dir
                new_ds.append(meta_info)

            datasets.append([])
            scene_num = 0
            new_seq = []
            for frame in ds:
                if frame["id"] == 0 and len(new_seq) > 0:
                    datasets[-1].append(new_seq)
                    self.max_seq_len = max(self.max_seq_len, len(new_seq))
                    new_seq = []
                    scene_num += 1
                new_seq.append(frame)
            if len(new_seq) > 0:
                datasets[-1].append(new_seq)
                self.max_seq_len = max(self.max_seq_len, len(new_seq))
                scene_num += 1
            self.max_seq_num_per_scene = max(self.max_seq_num_per_scene, scene_num)

        return datasets

    def fuse_and_pad_datasets(self, datasets):
        output = []
        for ds in datasets:
            ds = ds + list(reversed(ds))
            ds = (self.max_seq_len // len(ds) + 1) * ds
            ds = ds[: self.max_seq_len]
            output.append(ds)
        return output

    def load_meta_info_list(self, path):
        return pkl.load(open(path, "rb"))

    def valid_depth_ratio(self, depth):
        mask = depth > 1e-5
        ratio = mask.sum() / mask.size
        return ratio

    def load_seq(self, meta_info_list, base_dir):
        Tcw_tensor = []
        K_tensor = []
        depth_tensor = []
        last_name = None
        img_tensor = []
        self.transform.random_parameters()
        for idx, meta_info in enumerate(meta_info_list):
            if last_name is not None:
                a = meta_info["file_name"].split("/")[:2]
                if a[0] != last_name[0] or a[1] != last_name[1]:
                    print(a, last_name)

            last_name = meta_info["file_name"].split("/")[:2]

            img, depth, Tcw, K = load_one_img(
                base_dir,
                meta_info,
                self.lmdb_data_reader,
                read_img=True,
                H=self.img_H,
                W=self.img_W,
                dataset=self.dataset,
            )

            if self.crop_img_func is not None:
                img, K = self.crop_img_func(img, K)
            if self.crop_depth_func is not None:
                depth, K = self.crop_depth_func(depth, K)

            img, depth, Tcw, K = self.transform(img, depth, Tcw, K)

            Tcw_tensor.append(Tcw)
            K_tensor.append(K)

            depth_tensor.append(depth)
            img_tensor.append(img)

        Tcw_tensor = np.stack(Tcw_tensor).astype(np.float32)
        K_tensor = np.stack(K_tensor).astype(np.float32)
        depth_tensor = np.stack(depth_tensor).astype(np.float32)

        result = {
            "Tcw": Tcw_tensor,
            "K": K_tensor,
            "depth": depth_tensor,
            "img": np.stack(img_tensor).astype(np.float32).transpose(0, 3, 1, 2),
        }
        return result

    def get_next_frame(self, idx, meta_info, idxs):
        start_idx = max(self.start_idxs[idx], idx - self.tempo_interval)
        return start_idx

    def crop_img(self, img, K, new_K):
        return img

    def load_seq_scene(self, topk_meta_info_list, base_dir, q_Tcw, q_K, q_depth):
        Tcw_tensor = []
        K_tensor = []
        depth_tensor = []
        img_tensor = []
        delete_idxs = []
        for i in list(range(len(topk_meta_info_list))):
            meta_info = topk_meta_info_list[i]
            img, depth, Tcw, K = load_one_img(
                base_dir,
                meta_info,
                self.lmdb_data_reader,
                read_img=True,
                H=self.img_H,
                W=self.img_W,
                dataset=self.dataset,
            )

            if self.crop_img_func is not None:
                img, K = self.crop_img_func(img, K, False)

            if self.crop_depth_func is not None:
                depth, K = self.crop_depth_func(depth, K, False)

            img, depth, Tcw, K = self.transform(img, depth, Tcw, K)

            Tcw_tensor.append(Tcw)
            K_tensor.append(K)
            depth_tensor.append(depth)
            img_tensor.append(img)

            if len(Tcw_tensor) == self.ref_topk:
                break
        if self.pad_image and len(Tcw_tensor) < self.ref_topk:
            Tcw_tensor = Tcw_tensor + [Tcw_tensor[0]] * (
                self.ref_topk - len(Tcw_tensor)
            )
            K_tensor = K_tensor + [K_tensor[0]] * (self.ref_topk - len(K_tensor))
            depth_tensor = depth_tensor + [depth_tensor[0]] * (
                self.ref_topk - len(depth_tensor)
            )
            img_tensor = img_tensor + [img_tensor[0]] * (
                self.ref_topk - len(img_tensor)
            )

        for idx in reversed(delete_idxs):
            del topk_meta_info_list[idx]

        Tcw_tensor = np.stack(Tcw_tensor).astype(np.float32)
        K_tensor = np.stack(K_tensor).astype(np.float32)
        depth_tensor = np.stack(depth_tensor).astype(np.float32)

        result = {
            "Tcw": Tcw_tensor,
            "K": K_tensor,
            "depth": depth_tensor,
            "img": np.stack(img_tensor).astype(np.float32).transpose(0, 3, 1, 2),
        }

        return result, topk_meta_info_list

    def __getitem__(self, idx):
        p = 0
        q_result, r_result = None, None
        idx = self.valid_idx_list[idx]
        # idx=1847
        while r_result is None:
            if self.meta_info_list[idx]["topk"] is None:
                idx = self.remove_and_random_idx(idx)
                continue

            meta_info_list = []
            p += 1
            _idx = idx
            idxs = []
            random_temporal_len = self.tempo_len

            for i in range(random_temporal_len):
                idxs.append(_idx)
                meta_info_list.append(self.meta_info_list[_idx])
                _idx = self.get_next_frame(_idx, self.meta_info_list[_idx], idxs)

            q_result = self.load_seq(
                meta_info_list[: self.tempo_len], meta_info_list[0]["base_dir"]
            )

            if q_result is None:
                idx = self.remove_and_random_idx(idx)
                continue
            q_result["id"] = idxs

            r_result, self.meta_info_list[idx]["topk"] = self.load_seq_scene(
                self.meta_info_list[idx]["topk"],
                meta_info_list[0]["base_dir"],
                q_result["Tcw"][0],
                q_result["K"][0],
                q_result["depth"][0],
            )

            if r_result is None:
                idx = self.remove_and_randon_idx(idx)
                continue

        return q_result, r_result

    def __len__(self):
        return len(self.valid_idx_list)


class VideoDatasetScannet(VideoDataset):
    def __init__(self, cfg, transform):
        super().__init__(cfg, transform)
        self.dataset = "scannet"
        self.img_H, self.img_W = 240, 320

    def reset_valid_list(self):
        valid_idx_list = []
        # tot=0
        for i, meta_data in enumerate(self.meta_info_list):
            topk_list = meta_data["topk"]
            new_topk_list = []
            max_overlap = 0
            overlap = []
            for j in range(min(len(meta_data["overlap"]), len(topk_list))):
                meta_data_topk = topk_list[j]
                if self.overlap_check:
                    if meta_data["overlap"][j] > self.min_overlap_thresh:
                        new_topk_list.append(meta_data_topk)
                        overlap.append(meta_data["overlap"][j])
                        max_overlap = max(max_overlap, meta_data["overlap"][j])

            if self.overlap_check:
                if max_overlap > self.max_overlap_thresh:
                    new_topk_list = list(zip(overlap, new_topk_list))
                    new_topk_list = sorted(new_topk_list, key=lambda k: -k[0])
                    overlap = [k for k, v in new_topk_list]
                    new_topk_list = [v for k, v in new_topk_list]
                    meta_data["topk"] = new_topk_list
                    meta_data["overlap"] = overlap
                else:
                    meta_data["topk"] = []

            if not self.overlap_check or len(meta_data["topk"]) >= self.check_topk:
                valid_idx_list.append(i)

        return valid_idx_list


class VideoDataset7scene(VideoDataset):
    def __init__(self, cfg, transform):
        super().__init__(cfg, transform)

        self.depth_K = np.asarray(
            [[585, 0, 320], [0, 585, 240], [0, 0, 1]], dtype=np.float32
        )
        self.img_K = np.asarray(
            [[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32
        )
        self.crop_img_func = self.crop_img
        self.dataset = "7scene"

    def crop_img(self, img, K=None, no_none=None):
        return crop_by_intrinsic(img, self.img_K, self.depth_K), self.depth_K

    def reset_valid_list(self):
        valid_idx_list = []
        for i, meta_data in enumerate(self.meta_info_list):
            topk_list = meta_data["topk"]
            new_topk_list = []
            max_overlap = 0
            for j, meta_data_topk in enumerate(topk_list):

                if self.overlap_check:
                    if (
                        meta_data["overlap"][j] > self.min_overlap_thresh
                        and meta_data["depth_error"][j] < self.depth_error_thresh
                    ):
                        new_topk_list.append(meta_data_topk)
                        max_overlap = max(max_overlap, meta_data["overlap"][j])

                    if max_overlap > self.max_overlap_thresh:
                        meta_data["topk"] = new_topk_list
                    else:
                        meta_data["topk"] = []

            if len(meta_data["topk"]) >= self.ref_topk:
                valid_idx_list.append(i)

        return [
            valid_idx_list[i]
            for i in range(0, len(valid_idx_list), self.tempo_interval)
        ]


class VideoDatasetCambridge(VideoDataset):
    def __init__(self, cfg, transform):
        super().__init__(cfg, transform)

        self.depth_K = np.asarray(
            [[418.7109375, 0, 240], [0, 418.7109375, 135], [0, 0, 1]], dtype=np.float32
        )
        self.img_K = np.asarray(
            [[418.7109375, 0, 240], [0, 418.7109375, 135], [0, 0, 1]], dtype=np.float32
        )
        self.new_K = np.asarray(
            [[418.7109375, 0, 180], [0, 418.7109375, 135], [0, 0, 1]], dtype=np.float32
        )

        self.crop_img_func = self.crop_img if cfg.crop else None
        self.crop_depth_func = self.crop_depth if cfg.crop else None
        self.dataset = "cambridge"
        self.img_H, self.img_W = 270, 480

    def crop_img(self, img, K=None, query=False):
        return (
            crop_by_intrinsic(img, self.img_K.copy(), self.new_K.copy()),
            self.new_K.copy(),
        )

    def crop_depth(self, depth, K=None, query=False):
        return (
            crop_by_intrinsic(
                depth, self.depth_K.copy(), self.new_K.copy(), interp_method="nearest"
            ),
            self.new_K.copy(),
        )

    def reset_valid_list(self):
        valid_idx_list = []
        for i, meta_data in enumerate(self.meta_info_list):
            topk_list = meta_data["topk"]
            new_topk_list = []
            max_overlap = 0
            if self.overlap_check:
                overlaps = meta_data["overlap"]
            new_overlaps = []
            for j, meta_data_topk in enumerate(topk_list):

                if self.overlap_check:
                    if overlaps[j] > self.min_overlap_thresh and (
                        "depth_error" not in meta_data
                        or meta_data["depth_error"][j] < self.depth_error_thresh
                    ):
                        new_topk_list.append(meta_data_topk)
                        new_overlaps.append(overlaps[j])
                        max_overlap = max(max_overlap, overlaps[j])

                    if max_overlap > self.max_overlap_thresh:
                        meta_data["topk"] = new_topk_list
                        meta_data["overlap"] = new_overlaps
                    else:
                        meta_data["topk"] = []

            if not self.overlap_check or len(meta_data["topk"]) >= self.ref_topk:
                valid_idx_list.append(i)

        return [
            valid_idx_list[i]
            for i in range(0, len(valid_idx_list), self.tempo_interval)
        ]

