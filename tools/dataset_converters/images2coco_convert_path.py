import os
import json

with open("/ws/data/cityscapes/annotations/instancesonly_filtered_gtFine_demo_old.json", "r") as f:
    data = json.load(f)
for i, img in enumerate(data['images']):
    data['images'][i]['file_name'] = img['file_name'].replace("/ws/data/cityscapes/leftImg8bit/demo/", "")
with open("/ws/data/cityscapes/annotations/instancesonly_filtered_gtFine_demo.json", "w") as f:
    json.dump(data, f)
