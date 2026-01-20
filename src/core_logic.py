import numpy as np
import pandas as pd
import tifffile as tiff
import imageio.v3 as iio
from pathlib import Path
from scipy.ndimage import find_objects, center_of_mass


def read_image_folder(folder_path):
    if not folder_path or not Path(folder_path).exists():
        return None
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([f for f in Path(folder_path).glob("*") if f.suffix.lower() in exts])
    if not files:
        return None
    img_list = []
    for f in files:
        try:
            img = (
                tiff.imread(str(f))
                if f.suffix.lower() in {".tif", ".tiff"}
                else iio.imread(str(f))
            )
            img_list.append(img)
        except:
            continue
    return np.stack(img_list) if img_list else None


def scan_frame_for_stats(t, frame, stats, cents):
    uids = np.unique(frame)
    uids = uids[uids != 0]
    frame_ids = [int(u) for u in uids]
    objs = find_objects(frame.astype(np.int32))
    for uid in uids:
        uid = int(uid)
        if uid not in stats:
            stats[uid] = [t, t]
        else:
            stats[uid][1] = t
        sl = objs[uid - 1]
        if sl:
            cm = center_of_mass(frame[sl] == uid)
            cents[(t, uid)] = (cm[0] + sl[0].start, cm[1] + sl[1].start)
    return frame_ids
