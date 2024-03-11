#%%
# Purpose: Visualize bboxes and create train + test split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Rectangle
import os
from pathlib import Path
from collections import Counter
#%%
data_dir = "C:/Users/frank/Documents/GitHub/fighter-jets-photos/crop"
yolo_dir = "C:/Users/frank/Documents/GitHub/yolo-mil-air/datasets/"
os.chdir(yolo_dir)
class_names = sorted(os.listdir(data_dir))
image_dir = "C:/Users/frank/Documents/GitHub/yolo-mil-air/datasets/images"
label_dir = "C:/Users/frank/Documents/GitHub/yolo-mil-air/datasets/labels"
image_paths = os.listdir("C:/Users/frank/Documents/GitHub/yolo-mil-air/datasets/images")
label_paths = os.listdir("C:/Users/frank/Documents/GitHub/yolo-mil-air/datasets/labels")
#%% Index 2 class mappings
img_directory = "C:/Users/frank/Documents/GitHub/yolo-mil-air/datasets/images"
sample_ids = np.random.randint(0, len(image_paths), 10)
index2class = {class_index: class_name for class_index, class_name in enumerate(class_names)}
cmap = plt.get_cmap('rainbow', len(index2class))
# %%
fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(12, 4 * 7))

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
        print(rect)
        ax[i].add_patch(rect)
        ax[i].text(
            xmin, ymin, 
            index2class[int(class_id)], 
            ha='left', va='bottom',
            bbox={'facecolor': cmap(int(class_id)), 'alpha': 0.5}
        )

    ax[i].imshow(image)
    ax[i].axis('off')
# %% Train Test Split Creation
train_image_paths, val_image_paths = train_test_split(image_paths, train_size= 0.7, random_state= 42, shuffle=True)
val_image_paths, test_image_paths = train_test_split(val_image_paths, train_size = 0.5, random_state = 42, shuffle = True)
with open(f"{yolo_dir}/train_split.txt", 'w') as f:
    f.writelines(f'./images/{img}\n' for img in train_image_paths)
with open(f'{yolo_dir}/val_split.txt', 'w') as f:
    f.writelines(f'./images/{img}\n' for img in val_image_paths)
with open(f'{yolo_dir}/test_split.txt', 'w') as f:
    f.writelines(f'./{img}\n' for img in test_image_paths)

# %%
# %%
class_counter = {'train': Counter(), 'val': Counter()}
class_freqs = {}

with open('train_split.txt', 'r') as f:
    for line in f:
        image_id = line.split('/')[-1].split('.')[0]
        df = np.loadtxt(f'{label_dir}/{image_id}.txt',ndmin=2)
        class_counter['train'].update(df[:, 0].astype(int))
# get class freqs
total = sum(class_counter['train'].values())
class_freqs['train'] = {k: v / total for k, v in class_counter['train'].items()}
        
with open('val_split.txt', 'r') as f:
    for line in f:
        image_id = line.split('/')[-1].split('.')[0]
        df = np.loadtxt(f'{label_dir}/{image_id}.txt',ndmin=2)
        class_counter['val'].update(df[:, 0].astype(int))
# get class freqs
total = sum(class_counter['val'].values())
class_freqs['val'] = {k: v / total for k, v in class_counter['val'].items()}
# %%
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(range(40), [class_freqs['train'][i] for i in range(40)], color='navy', label='train');
ax.plot(range(40), [class_freqs['val'][i] for i in range(40)], color='tomato', label='val');
ax.legend();
ax.set_xlabel('Class ID');
ax.set_ylabel('Class Frequency');
# %%
