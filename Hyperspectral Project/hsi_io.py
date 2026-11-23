"""
hsi_io.py  --  one loader for every scene of the project.

    from hsi_io import load
    d = load("synthetic")        # or "indian_pines", "paviaU", "salinas"
    d["cube"]        (H, W, B) float, scaled to ~[0, 1]
    d["wavelengths"] (B,) nm  (approximate for the real scenes)
    d["labels"]      (H, W) int, 0 = unlabelled for real scenes
    d["names"]       class names (index = label)
    d["endmembers"], d["abundances"], d["noise_sigma"]  only for "synthetic"
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")          # every script saves its figures here
os.makedirs(FIG, exist_ok=True)

INDIAN_PINES_CLASSES = ["unlabelled", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
                        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
                        "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
                        "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
PAVIA_CLASSES = ["unlabelled", "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
                 "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"]


def _mat(name, key_hint):
    from scipy.io import loadmat
    m = loadmat(os.path.join(DATA, name))
    key = [k for k in m if not k.startswith("__") and key_hint in k.lower()][0]
    return np.asarray(m[key])


def load(scene="synthetic"):
    if scene == "synthetic":
        path = os.path.join(DATA, "synthetic_cube.npz")
        if not os.path.exists(path):
            raise FileNotFoundError("run  python make_synthetic_cube.py  first")
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    if scene == "indian_pines":
        cube = _mat("Indian_pines_corrected.mat", "indian").astype(float)
        labels = _mat("Indian_pines_gt.mat", "gt").astype(int)
        wl, names = np.linspace(400, 2500, cube.shape[2]), INDIAN_PINES_CLASSES
    elif scene == "paviaU":
        cube = _mat("PaviaU.mat", "pavia").astype(float)
        labels = _mat("PaviaU_gt.mat", "gt").astype(int)
        wl, names = np.linspace(430, 860, cube.shape[2]), PAVIA_CLASSES
    elif scene == "salinas":
        cube = _mat("Salinas_corrected.mat", "salinas").astype(float)
        labels = _mat("Salinas_gt.mat", "gt").astype(int)
        wl, names = np.linspace(400, 2500, cube.shape[2]), [f"class {i}" for i in range(17)]
    else:
        raise ValueError(scene)
    cube /= np.percentile(cube, 99.5)               # digital numbers -> ~[0, 1]
    return dict(cube=cube, wavelengths=wl, labels=labels, names=np.array(names))


def band(wl, target_nm):
    """Index of the band closest to a wavelength."""
    return int(np.argmin(np.abs(np.asarray(wl) - target_nm)))


def rgb(cube, wl):
    """True-colour composite (R 650, G 550, B 470 nm), stretched for display."""
    img = cube[..., [band(wl, 650), band(wl, 550), band(wl, 470)]].astype(float)
    lo, hi = np.percentile(img, (2, 98))
    return np.clip((img - lo) / (hi - lo), 0, 1)
