"""
get_hyperspectral_data.py  --  download classic, free hyperspectral benchmark scenes.

Source: Grupo de Inteligencia Computacional, Universidad del Pais Vasco (EHU)
        https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

  indian_pines : AVIRIS, Indiana (USA) farmland, 145x145 px, 200 bands (400-2500 nm), 16 classes  ~6 MB
  paviaU       : ROSIS, University of Pavia (Italy!), 610x340 px, 103 bands (430-860 nm), 9 classes  ~33 MB
  salinas      : AVIRIS, Salinas Valley (USA), 512x217 px, 204 bands, 16 classes  ~26 MB

Run:   python get_hyperspectral_data.py                 (Indian Pines only, the smallest)
       python get_hyperspectral_data.py paviaU salinas  (more scenes)
If a link fails (the site sometimes moves files), open the page above in a browser,
download the .mat files by hand and put them in the data/ folder with the same names.
"""
import os
import sys
import urllib.request

BASE = "https://www.ehu.eus/ccwintco/uploads/"
SCENES = {
    "indian_pines": [("6/67/Indian_pines_corrected.mat", "Indian_pines_corrected.mat"),
                     ("c/c4/Indian_pines_gt.mat", "Indian_pines_gt.mat")],
    "paviaU": [("e/ee/PaviaU.mat", "PaviaU.mat"),
               ("5/50/PaviaU_gt.mat", "PaviaU_gt.mat")],
    "salinas": [("a/a3/Salinas_corrected.mat", "Salinas_corrected.mat"),
                ("f/fa/Salinas_gt.mat", "Salinas_gt.mat")],
}


def download(scene):
    os.makedirs("data", exist_ok=True)
    for path, name in SCENES[scene]:
        out = os.path.join("data", name)
        if os.path.exists(out):
            print(f"  {name}: already there")
            continue
        print(f"  downloading {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(BASE + path, out)
            print(f"{os.path.getsize(out)/1e6:.1f} MB")
        except Exception as e:                       # keep going, say what failed
            print(f"FAILED ({e}).  Get it by hand from the EHU page (see top of this file).")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["indian_pines"]):
        print(s)
        download(s)
