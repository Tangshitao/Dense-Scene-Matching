import lmdb
import sys
import numpy as np
import cv2
import os.path as osp
import os
import shutil
from skimage.io import imread


class LMDBModel:

    # Path to the LMDB
    lmdb_path = None

    # LMDB Environment handle
    __lmdb_env__ = None

    # LMDB context handle
    __lmdb_txn__ = None

    # LMDB Cursor for navigating data
    __lmdb_cursor__ = None

    """ Constructor and De-constructor
    """

    def __init__(self, lmdb_path, workers=3):
        self.lmdb_path = lmdb_path
        self.__start_session__(workers=workers)

    def __del__(self):
        self.close_session()

    """ Session Function
    """

    def __start_session__(self, workers):

        # Open LMDB file
        self.__lmdb_env__ = lmdb.open(
            self.lmdb_path, max_readers=workers, readonly=True
        )

        # Crete context
        self.__lmdb_txn__ = self.__lmdb_env__.begin(write=False)

        # Get the cursor of current lmdb
        self.__lmdb_cursor__ = self.__lmdb_txn__.cursor()

    def close_session(self):
        if self.__lmdb_env__ is not None:
            self.__lmdb_env__.close()
            self.__lmdb_env__ = None

    """ Read Routines
    """

    def read_by_key(self, key):

        """
        Read value in lmdb by providing the key
        :param key: the string that corresponding to the value
        :return: array data
        """
        value = self.__lmdb_cursor__.get(key.encode())
        return value

    def read_ndarray_by_key(self, key, dtype=np.float32):
        value = self.__lmdb_cursor__.get(key.encode())
        return np.fromstring(value, dtype=dtype)

    def len_entries(self):
        length = self.__lmdb_txn__.stat()["entries"]
        return length

    """ Static Utilities
    """

    @staticmethod
    def convert_to_img(data):

        """
        Transpose the data from the Caffe's format to the normal format
        :param data: ndarray object with dimension of (3, h, w)
        :return: transposed ndarray with dimension of (h, w, 3)
        """
        return data.transpose((1, 2, 0))

    def get_keys(self):
        keys = []
        for key, value in self.__lmdb_cursor__:
            keys.append(key)
        return keys


class LMDBWriter:

    """ Write the dataset to LMDB database
    """

    """ Variables
    """
    __key_counts__ = 0

    # LMDB environment handle
    __lmdb_env__ = None

    # LMDB context handle
    __lmdb_txn__ = None

    # LMDB Path
    lmdb_path = None

    """ Functions
    """

    def __init__(self, lmdb_path, auto_start=True):
        self.lmdb_path = lmdb_path
        self.__del_and_create__(lmdb_path)
        if auto_start is True:
            self.__start_session__()

    def __del__(self):
        self.close_session()

    def __del_and_create__(self, lmdb_path):
        """
        Delete the exist lmdb database and create new lmdb database.
        """
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        os.mkdir(lmdb_path)

    def __start_session__(self):
        self.__lmdb_env__ = lmdb.Environment(self.lmdb_path, map_size=1099511627776)
        self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True)

    def close_session(self):
        if self.__lmdb_env__ is not None:
            self.__lmdb_txn__.commit()
            self.__lmdb_env__.close()
            self.__lmdb_env__ = None
            self.__lmdb_txn__ = None

    def write_str(self, key, str):
        """
        Write the str data to the LMDB
        :param key: key in string type
        :param array: array data
        """
        # Put to lmdb
        self.__key_counts__ += 1
        self.__lmdb_txn__.put(key.encode(), str)
        if self.__key_counts__ % 10000 == 0:
            self.__lmdb_txn__.commit()
            self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True, buffers=True)

    def write_array(self, key, array):
        """
        Write the array data to the LMDB
        :param key: key in string type
        :param array: array data
        """
        # Put to lmdb
        self.__key_counts__ += 1
        self.__lmdb_txn__.put(key.encode(), array.tostring())
        if self.__key_counts__ % 10000 == 0:
            self.__lmdb_txn__.commit()
            self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True, buffers=True)


def load_extrinsic(meta_info):
    if len(meta_info["extrinsic_Tcw"]) == 16:
        Tcw = np.array(meta_info["extrinsic_Tcw"]).reshape(4, 4)
        Tcw = Tcw[:3, :]
    else:
        Tcw = np.array(meta_info["extrinsic_Tcw"]).reshape(3, 4)
    return Tcw


def load_intrinsic(meta_info):
    # if dataset=='default':
    K_param = meta_info["camera_intrinsic"]
    K = np.zeros((3, 3))
    K[0, 0] = K_param[0]
    K[1, 1] = K_param[1]
    K[2, 2] = 1
    K[0, 2] = K_param[2]
    K[1, 2] = K_param[3]
    return K


def load_depth_from_png(tiff_file_path):
    depth = cv2.imread(tiff_file_path, cv2.IMREAD_ANYDEPTH)
    depth[depth==65535]=0
    return depth


def load_one_img(
    base_dir, meta_info, lmdb_reader=None, H=480, W=640, read_img=True, dataset="7scene"
):
    Tcw = load_extrinsic(meta_info)
    K = load_intrinsic(meta_info)
    H = int(K[1, 2] * 2)
    W = int(K[0, 2] * 2)

    file_name = meta_info["file_name"]
    depth_file_name = meta_info["depth_file_name"]

    img = None

    if lmdb_reader is not None:
        if read_img:
            img = lmdb_reader.read_ndarray_by_key(file_name, dtype=np.uint8).reshape(
                H, W, 3
            )
        if dataset == "cambridge":
            dtype = np.float32
        else:
            dtype = np.uint16
        depth = lmdb_reader.read_ndarray_by_key(depth_file_name, dtype=dtype).reshape(
            H, W
        )
        depth_file_name = osp.join(base_dir, depth_file_name)
        
    else:
        img_path = osp.join(base_dir, file_name)
        depth_file_name = osp.join(base_dir, depth_file_name)
        if read_img:
            img = imread(img_path)
        depth = load_depth_from_png(depth_file_name)

    depth=depth.astype(np.float32)/1000
    depth[depth < 1e-5] = 0
    return img, depth, Tcw, K
