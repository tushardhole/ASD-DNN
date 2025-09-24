import numpy as np
from nilearn.datasets import fetch_abide_pcp
from nilearn.image import resample_img
import nibabel as nib

def preprocess_fmri(img_path, target_shape=(32,32,32)):
    img = nib.load(img_path)
    resampled_img = resample_img(img,
                                 target_affine=np.eye(4),
                                 target_shape=target_shape)
    data = resampled_img.get_fdata()  # X x Y x Z x T
    mean_vol = data.mean(axis=3)
    std_vol = data.std(axis=3)
    return np.stack([mean_vol, std_vol], axis=0)

def fetch_abide_subjects(n_subjects=3):
    abide = fetch_abide_pcp(n_subjects=n_subjects)
    subjects = []
    for i in range(n_subjects):
        func_path = abide.func_preproc[i]
        data = preprocess_fmri(func_path)
        label = int(abide.phenotypic.iloc[i]['DX_GROUP'])
        subjects.append({'data': data, 'label': label})
    return subjects
