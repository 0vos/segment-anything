import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# step 2
image = cv2.imread('images/IMG_8797.JPG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()
start_ts = time.time()
print("time start: ", time.strftime("%H:%M:%S", time.localtime(start_ts)))
masks = mask_generator.generate(image)
end_ts = time.time()
print(len(masks))
print(masks[0].keys())
print("time end: ", time.strftime("%H:%M:%S", time.localtime(end_ts)))
print(f"elapsed seconds: {end_ts - start_ts:.3f}")
print("")
# step 3
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
# step 4
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 
