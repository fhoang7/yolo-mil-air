#%%
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Rectangle
import os
from pathlib import Path
#%%
data_dir = "C:/Users/frank/OneDrive/Documents/Data Projects/fighter-jets-photos/crop"
yolo_dir = "C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air/datasets/"
os.chdir(yolo_dir)
class_names = sorted(os.listdir(data_dir))
image_dir = "C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air/datasets/images"
label_dir = "C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air/datasets/labels"
image_paths = os.listdir("C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air/datasets/images")
label_paths = os.listdir("C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air/datasets/labels")
#%% Index 2 class mappings
img_directory = "C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air/datasets/images"
sample_ids = np.random.randint(0, len(image_paths), 4)
index2class = {class_index: class_name for class_index, class_name in enumerate(class_names)}
cmap = plt.get_cmap('rainbow', len(index2class))
# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 4 * 7))

for i, sample_id in enumerate(sample_ids):
    # load image and bboxes
    image = np.array(Image.open(f"{image_dir}/{image_paths[sample_id]}"))
    bboxes = np.loadtxt(f"{label_dir}/{label_paths[sample_id]}", ndmin=2)
    # get image shape
    image_h, image_w = image.shape[:2]

    for bbox in bboxes:
        class_id, xc, yc, w, h = bbox
        # rescale to image size
        xc, yc, w, h = image_w * xc, image_h * yc, image_w * w, image_h * h
        xmin, ymin = xc - w / 2, yc - h / 2
        rect = Rectangle(
            (xmin, ymin), 
            w, h, 
            linewidth=4, 
            edgecolor=cmap(int(class_id)), 
            facecolor='none',
            alpha=0.5
        )
        ax[i].add_patch(rect)
        ax[i].text(
            xmin, ymin, 
            index2class[int(class_id)], 
            ha='left', va='bottom',
            bbox={'facecolor': cmap(int(class_id)), 'alpha': 0.5}
        )

    ax[i].imshow(image)
    ax[i].axis('off');
# %%
