"""
Created on 16:05, Apr. 5th, 2021
Author: fassial
Filename: stimulus.py
"""
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "stimulus",
    "stim_params",
]

class stimulus(object):

    @staticmethod
    def get(stim_params):
        # get corresponding func in stimulus
        try:
            func = getattr(stimulus, "_{}".format(stim_params.name))
        except Exception:
            raise ValueError("ERROR: Unknown function in stimulus.get.")
        # gen arr, idxs, stim
        arr, idxs, stim = func(
            height = stim_params.height,
            width = stim_params.width,
            stim_intensity = stim_params.intensity,
            stim_params = stim_params.others
        )
        return arr, idxs, stim

    @staticmethod
    def show(stim_params, target = "img", img_fname = None):
        # get corresponding arr & idxs & stim
        arr, _, stim = stimulus.get(stim_params = stim_params)

        # set img
        if target == "img":
            img = arr.reshape((stim_params.height, stim_params.width))
        elif target == "stim":
            img = stim.reshape((stim_params.height, stim_params.width))
            img_max, img_min = img.max(), img.min()
            img = 1 - (img - img_min) / img_max
        else:
            raise ValueError("ERROR: Unknown target in stimulus.show.")

        # gen plt
        plt.imshow(img[::-1,:], cmap = "gray")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad = 1.)

        # save img
        if img_fname is None:
            plt.show()
        else:
            plt.savefig(fname = img_fname)

    """
    tool funcs
    """
    @staticmethod
    def _dist(i, j, center):
        return np.power((i - center[0]) ** 2 + (j - center[1]) ** 2, 0.5)

    """
    stimulus funcs
    """
    # black stimulus funcs
    @staticmethod
    def _black(height = 50, width = 50, stim_intensity = [15,], **kwargs):
        # init arr
        arr = np.zeros((height, width), dtype=np.float32)

        # reshape arr & set stim
        arr = arr.reshape((height * width,))
        idxs = [np.where(arr == _id)[0] for _id in [0.,]]
        stim = arr * stim_intensity[0]
        return arr, idxs, stim

    # black stimulus funcs
    @staticmethod
    def _white(height = 50, width = 50, stim_intensity = [15,], stim_params = {
        "noise": 0,
    }):
        # init arr
        arr = np.ones((height, width), dtype=np.float32)

        # reshape arr & set stim
        arr = arr.reshape((height * width,))
        idxs = [np.where(arr == _id)[0] for _id in [1.,]]
        stim = arr * stim_intensity[0]
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )
        return arr, idxs, stim

    # cross stimulus funcs
    @staticmethod
    def _cross(height = 50, width = 50, stim_intensity = [12, 20], stim_params = {
        "length": 10,
        "lw": 5,
        "noise": 0,
    }):
        # init arr & w_center & h_center
        arr = np.zeros((height, width), dtype = np.float32)
        w_center = width // 2
        h_center = height // 2

        ## set arr
        # set arr's row
        row_i = height // 2 - stim_params["lw"] // 2
        for col_i in range(w_center - stim_params["length"], w_center + stim_params["length"]):
            for i in range(stim_params["lw"]):
                arr[row_i + i, col_i] = 1.
        # set arr's col
        col_i = w_center - stim_params["lw"] // 2
        for row_i in range(h_center - stim_params["length"], h_center + stim_params["length"]):
            for i in range(stim_params["lw"]):
                arr[row_i, col_i + i] = 1.

        # reshape arr & set stim
        arr = arr.reshape((height * width,))
        idxs = [np.where(arr == _id)[0] for _id in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(len(stim_intensity)):
            stim[idxs[i]] = stim_intensity[i]
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )
        return 1 - arr, idxs, stim

    # circle stimulus funcs
    @staticmethod
    def _circle(height = 50, width = 50, stim_intensity = [10, 15], stim_params = {
        "radius": 15,
        "noise": 0,
    }):
        # init arr
        arr = np.zeros((height, width), dtype=np.float32)

        # set arr
        for i in range(height):
            for j in range(width):
                if stimulus._dist(i, j, (height / 2, width / 2)) < stim_params["radius"]:
                    arr[i, j] = 1.

        # reshape arr & set stim
        arr = arr.reshape((height * width,))
        idxs = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(len(stim_intensity)):
            stim[idxs[i]] = stim_intensity[i]
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )
        return 1 - arr, idxs, stim

    @staticmethod
    def _one_hole(height = 50, width = 50, stim_intensity = [15, 20, 15], stim_params = {
        "inner_radius": 15,
        "outer_radius": 20,
        "position": "center", # ["center", "corner", "line_middle"]
        "noise": 0,
    }):
        # init arr & idxx
        arr = np.zeros((height, width), dtype=np.float32)
        idx1, idx2, idx3 = [], [], []

        # set center
        if stim_params["position"] == "center":
            center = (height / 2, width / 2)
        elif stim_params["position"] == "corner":
            center = (stim_params["outer_radius"] - 1, stim_params["outer_radius"] - 1)
        elif stim_params["position"] == "line_middle":
            center = (stim_params["outer_radius"] - 1, width / 2)
        else:
            raise ValueError("ERROR: Unknown stim_params[\"position\"] in stimulus._one_hole.")

        for i in range(height):
            for j in range(width):
                dist = stimulus._dist(i, j, center)
                if dist < stim_params["inner_radius"]:
                    idx1.append(i * width + j)
                    arr[i, j] = 0.
                elif stim_params["inner_radius"] <= dist < stim_params["outer_radius"]:
                    idx2.append(i * width + j)
                    arr[i, j] = 1.
                else:
                    idx3.append(i * width + j)
                    arr[i, j] = 0.

        # reshape arr & set stim
        arr = arr.reshape((height * width,))
        idxs = [idx1, idx2, idx3]
        stim = np.zeros_like(arr)
        for i in range(len(stim_intensity)):
            stim[idxs[i]] = stim_intensity[i]
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )
        return 1 - arr, idxs, stim

    @staticmethod
    def _two_holes(height = 50, width = 50, stim_intensity = [15, 15, 15, 20], stim_params = {
        "inner_radius": 8,
        "outer_radius": 24,
        "noise": 0,
    }):
        # check r_i * 2 <= r_o
        assert stim_params["inner_radius"] * 2 <= stim_params["outer_radius"]

        # init arr & idxx & centerx
        arr = np.zeros((height, width), dtype=np.float32)
        idx1, idx2, idx3, idx4 = [], [], [], [] # bg, hole 1, hole 2, black obj
        center0 = (height // 2, width // 2)
        center1 = (height // 2 - stim_params["outer_radius"] // 2, width // 2)
        center2 = (height // 2 + stim_params["outer_radius"] // 2, width // 2)

        # set arr
        for i in range(height):
            for j in range(width):
                dist0 = stimulus._dist(i, j, center0)
                dist1 = stimulus._dist(i, j, center1)
                dist2 = stimulus._dist(i, j, center2)
                if dist1 < stim_params["inner_radius"]:
                    idx2.append(i * width + j)
                    arr[i, j] = 0.
                elif dist2 < stim_params["inner_radius"]:
                    idx3.append(i * width + j)
                    arr[i, j] = 0.
                elif dist0 < stim_params["outer_radius"]:
                    idx4.append(i * width + j)
                    arr[i, j] = 1.
                else:
                    idx1.append(i * width + j)
                    arr[i, j] = 0.

        # reshape arr & set stim
        arr = arr.reshape((height * width,))
        idxs = [idx1, idx2, idx3, idx4]
        stim = np.zeros_like(arr)
        for i in range(len(stim_intensity)):
            stim[idxs[i]] = stim_intensity[i]
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )
        return 1 - arr, idxs, stim

class stim_params:

    def __init__(self, name, height, width, intensity, others):
        # init params
        self.name = name
        self.height = height
        self.width = width
        self.intensity = intensity
        self.others = others

default_stim_params = {
    "white": stim_params(
        name = "white",
        height = 50,
        width = 50,
        intensity = [15,],
        others = {
            "noise": 0,
        }
    ),
    "black": stim_params(
        name = "black",
        height = 50,
        width = 50,
        intensity = [15,],
        others = {
            "noise": 0,
        }
    ),
    "cross": stim_params(
        name = "cross",
        height = 50,
        width = 50,
        intensity = [12, 20],
        others = {
            "length": 10,
            "lw": 5,
            "noise": 0,
        }
    ),
    "circle": stim_params(
        name = "circle",
        height = 50,
        width = 50,
        intensity = [10, 15],
        others = {
            "radius": 15,
            "noise": 0,
        }
    ),
    "one_hole": stim_params(
        name = "one_hole",
        height = 50,
        width = 50,
        intensity = [15, 20, 15],
        others = {
            "inner_radius": 15,
            "outer_radius": 20,
            "position": "center",
            "noise": 0,
        }
    ),
    "two_holes": stim_params(
        name = "two_holes",
        height = 50,
        width = 50,
        intensity = [15, 15, 15, 20],
        others = {
            "inner_radius": 8,
            "outer_radius": 24,
            "noise": 0,
        }
    ),
}

if __name__ == "__main__":
    stimulus.show(default_stim_params["two_holes"])

