#!/usr/bin/env python

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import imgviz
from labelme import utils

def convert_avm_to_labelme(avm_json_file):
    with open(avm_json_file, 'r') as f:
        avm_data = json.load(f)

    # 假设avm_data的结构与labelme相似
    shapes = []
    for item in avm_data['shapes']:
        shape = {
            "label": item['label'],
            "points": item['points'],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    return shapes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()

    # 转换avm格式为labelme格式
    shapes = convert_avm_to_labelme(args.json_file)

    # 这里假设您有一个图像数据
    img = np.zeros((256, 256, 3), dtype=np.uint8)  # 示例图像，您需要替换为实际图像数据
    label_name_to_value = {"_background_": 0}
    
    for shape in sorted(shapes, key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _ = utils.shapes_to_label(img.shape, shapes, label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(
        lbl,
        imgviz.asgray(img),
        label_names=label_names,
        font_size=30,
        loc="rb",
    )

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl_viz)
    plt.show()

if __name__ == "__main__":
    main()
