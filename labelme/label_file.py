import base64
import contextlib
import io
import orjson
import os.path as osp
import uuid
import random

import os
import numpy as np
import cv2

import PIL.Image
from loguru import logger

from labelme import PY2
from labelme import __version__
from labelme import utils



PIL.Image.MAX_IMAGE_PIXELS = None
# 颜色字典（假设 utils.cx_color_dict_OurVersion 已定义）
cx_color_dict_OurVersion = {
    "Background": [70, 70, 70],
    "Road": [128, 64, 128],
    "Lane_line": [240, 240, 240],
    "Parking_line": [70, 120, 120],
    "Parking_slot": [70, 12, 120],
    "Arrow": [70, 120, 12],
    "Crosswalk_line": [0, 120, 120],
    "No_parking_sign_line": [200, 120, 120],
    "Speed_bump": [70, 200, 120],
    "Parking_lock_open": [70, 120, 200],
    "Parking_lock_closed": [100, 0, 0],
    "Traffic_cone": [130, 100, 10], # 1
    "Parking_rod": [250, 120, 120],
    "Limiter_pole": [70, 0, 250],
    "Pillar": [250, 170, 160], # 1
    "Immovable_obstacle": [150, 120, 90], # 1
    "Person": [220, 20, 60],
    "Car": [0, 0, 142],
    "self_car": [20, 180,80],
    "Curb":  [140, 100, 100],
    "Movable_obstacle": [230, 150, 140], #1
    "Guide_line": [160, 160, 160],
    "Center_lane": [170, 10, 10],
    "Cover": [170, 170, 170],
    "sewer": [100, 20, 10]
}
type_colors = cx_color_dict_OurVersion
# 将 RGB 值归一化到 0 到 1 之间
type_colors = {k: [v[2], v[1], v[0]] for k, v in type_colors.items()}  # OpenCV 使用 BGR

need_show_list = ["Background", "Road", "Person", "Traffic_cone", "Car", "Movable_obstacle", "Immovable_obstacle", "Lane_line", "Parking_line", "Parking_slot", "Arrow", "Crosswalk_line", "No_parking_sign_line", "Speed_bump", "Parking_lock_open", "Parking_lock_closed", "Parking_rod", "Limiter_pole", "Curb", "Guide_line", "Center_lane", "Cover", "sewer", "self_car"]

# @contextlib.contextmanager
# def open(name, mode):
#     assert mode in ["r", "w", "rb"]
#     if PY2:
#         mode += "b"
#         encoding = None
#     else:
#         encoding = "utf-8"
#     yield io.open(name, mode, encoding=encoding)
#     return


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        try:
            # 检查目录名称
            dir_name = osp.basename(osp.dirname(osp.dirname(filename)))
            # print("===============label_file.load.dir_name=============", dir_name)
            if dir_name.startswith("Slot_"):
                return self.load_slot(filename)
            
            if dir_name.startswith("2D-OD_"):
                return self.load_2dod(filename)

            with open(filename, "rb") as f:  # 以二进制模式打开文件
                data = orjson.loads(f.read())  # 使用 orjson 读取数据

            # 解析avm_gdc格式的JSON
            if "anno" in data:
                annotations = data["anno"]
                self.shapes = []
                for annotation in annotations:
                    object_type = annotation["category"]["type"]
                    # 只加载 Parking_slot 或 Parking_line 类型的注释
                    if object_type not in need_show_list:
                        continue
                    # 假设有 allPointsY 字段
                    allPointsX = annotation["data"]["allPointsX"]
                    allPointsY = annotation["data"]["allPointsY"]
                    
                    # 将 x 和 y 点对组合成一个列表
                    points = [(x, y) for x, y in zip(allPointsX, allPointsY)]

                    shape = {
                        "label": annotation["category"]["type"],
                        "points": points,
                        "shape_type": "polygon",  # 根据需要调整
                        "flags": {},
                        "description": "{0}+{1}+{2}".format(list(annotation["attrs"].keys())[0], list(annotation["attrs"].values())[0], annotation["category"]["child"]["attributes"]["Attribute"]),  # 根据需要添加描述
                        "group_id": None,  # 根据需要添加组 ID
                        "mask": None,  # 如果有 mask 数据，可以在这里处理
                        "other_data": {},
                    }
                    self.shapes.append(shape)
                    # break

            otherData = {}
            self.otherData = otherData
            self.flags = {}

            # 获取JSON文件的上一级文件夹
            parent_dir = osp.dirname(osp.abspath(filename))
            avm_dir = osp.join(osp.dirname(parent_dir), "AVM")  # 上一级文件夹下的AVM文件夹
            self.segPath = osp.join(avm_dir, osp.splitext(osp.basename(filename))[0] + ".jpg")  # 构建新的JPG文件路径

            # 加载语义分割图数据
            seg_dir = osp.join(osp.dirname(parent_dir), "vis_avm")  # 上一级文件夹下的vis_avm文件夹
            # self.imagePath = osp.join(seg_dir, osp.splitext(osp.basename(filename))[0][:-7] + "seg.png")  # 构建新的JPG文件路径
            self.imagePath = osp.join(seg_dir, osp.splitext(osp.basename(filename))[0] + ".png")  # 构建新的JPG文件路径

            # 加载图像数据
            # 加载图像数据
            original_image = PIL.Image.open(self.imagePath)
            width, height = original_image.size
            # 裁剪右半部分
            right_half = original_image.crop((width // 2, 0, width, height))
            # 将裁剪后的图像转换为适合存储的格式
            with io.BytesIO() as f:
                right_half.save(f, format='PNG')  # 保存为 PNG 格式
                f.seek(0)
                self.imageData = f.read()  # 读取裁剪后的图像数据
            # self.imageData = self.load_image_file(self.imagePath)
            self.segData = self.load_image_file(self.segPath)

            self.filename = filename

        except Exception as e:
            raise LabelFileError(e)

    def load_slot(self, filename):
        try:
            with open(filename, "rb") as f:
                data = orjson.loads(f.read())

            self.shapes = []
            
            # 处理所有标注
            for annotation in data["anno"]:
                object_type = annotation["category"]["type"]
                
                # 处理线类型标注
                if object_type in ["line", "entrance_line"]:
                    allPointsX = annotation["data"]["allPointsX"]
                    allPointsY = annotation["data"]["allPointsY"]
                    points = [(x, y) for x, y in zip(allPointsX, allPointsY)]
                    
                    shape = {
                        "label": object_type,
                        "points": points,
                        "shape_type": "line",
                        "flags": {},
                        "description": "",
                        "group_id": None,
                        "mask": None,
                        "other_data": {},
                    }
                    self.shapes.append(shape)
                
                # 处理点类型标注
                elif object_type == "keypoint":
                    # print(annotation["data"])
                    x = annotation["data"]["x"]
                    y = annotation["data"]["y"]
                    points = [(x, y)]
                    
                    shape = {
                        "label": "keypoint",
                        "points": points,
                        "shape_type": "point",
                        "flags": {},
                        "description": "",
                        "group_id": None,
                        "mask": None,
                        "other_data": {},
                    }
                    self.shapes.append(shape)
                
                # 处理自车标注
                elif object_type == "self_vehicle":
                    allPointsX = annotation["data"]["allPointsX"]
                    allPointsY = annotation["data"]["allPointsY"]
                    points = [(x, y) for x, y in zip(allPointsX, allPointsY)]
                    
                    shape = {
                        "label": "self_car",
                        "points": points,
                        "shape_type": "polygon",
                        "flags": {},
                        "description": "",
                        "group_id": None,
                        "mask": None,
                        "other_data": {},
                    }
                    self.shapes.append(shape)

            # 处理其他数据
            self.otherData = {}
            self.flags = {}

            # 获取图像路径
            parent_dir = osp.dirname(osp.abspath(filename))
            avm_dir = osp.join(osp.dirname(parent_dir), "AVM")
            self.segPath = osp.join(avm_dir, osp.splitext(osp.basename(filename))[0] + ".jpg")

            # 加载语义分割图数据
            seg_dir = osp.join(osp.dirname(parent_dir), "vis_avm")
            self.imagePath = osp.join(seg_dir, osp.splitext(osp.basename(filename))[0] + ".png")

            # 加载图像数据
            original_image = PIL.Image.open(self.imagePath)
            width, height = original_image.size
            right_half = original_image.crop((width // 2, 0, width, height))
            with io.BytesIO() as f:
                right_half.save(f, format='PNG')
                f.seek(0)
                self.imageData = f.read()
            self.segData = self.load_image_file(self.segPath)

            self.filename = filename

        except Exception as e:
            raise LabelFileError(e)
        
    def load_2dod(self, filename):
        try:
            with open(filename, "rb") as f:
                data = orjson.loads(f.read())

            self.shapes = []
            
            # 处理所有标注
            for annotation in data["anno"]:
                object_type = annotation["category"]["type"]
                
                # 提取 x, y, width, height
                x = annotation["data"]["x"]
                y = annotation["data"]["y"]
                width = annotation["data"]["width"]
                height = annotation["data"]["height"]

                # 计算矩形的四个角点
                points = [
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height)
                ]

                shape = {
                    "label": object_type,
                    "points": points,
                    "shape_type": "polygon",  # 根据需要调整
                    "flags": {},
                    "description": "{0}+{1}+{2}+{3}+{4}".format(list(annotation["attrs"].keys())[0], list(annotation["attrs"].values())[0], annotation["fileMetaUuid"], annotation["id"], annotation["objectId"]),  # 根据需要添加描述
                    "group_id": None,
                    "mask": None,  # 如果有 mask 数据，可以在这里处理
                    "other_data": {},
                }
                self.shapes.append(shape)

            # 处理其他数据
            self.otherData = {}
            self.flags = {}

            # 获取图像路径
            parent_dir = osp.dirname(osp.abspath(filename))
            avm_dir = osp.join(osp.dirname(parent_dir), "image")
            self.segPath = osp.join(avm_dir, osp.splitext(osp.basename(filename))[0] + ".jpg")

            # 加载语义分割图数据
            seg_dir = osp.join(osp.dirname(parent_dir), "vis_avm")
            self.imagePath = osp.join(seg_dir, osp.splitext(osp.basename(filename))[0] + ".png")

            # 加载图像数据
            original_image = PIL.Image.open(self.imagePath)
            with io.BytesIO() as f:
                original_image.save(f, format='PNG')
                f.seek(0)
                self.imageData = f.read()
            self.segData = self.load_image_file(self.segPath)

            self.filename = filename

        except Exception as e:
            raise LabelFileError(e)
        

    def draw_seg(self, json_path):      
        # 读取 JSON 文件
        with open(json_path, "rb") as f:  # 以二进制模式打开文件
            data = orjson.loads(f.read())  # 使用 orjson 读取数据
        
        # 创建一个空白图像（896x896，3通道，背景为白色）
        img = np.ones((896, 896, 3), dtype=np.uint8) * 255

        # Step1: 先画 Road
        for anno in data["anno"]:
            object_type = anno["category"]["type"]
            if object_type == "Road":
                all_points_x = anno["data"]["allPointsX"]
                all_points_y = anno["data"]["allPointsY"]
                points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                
                color = type_colors.get(object_type, (0, 0, 0))  # 默认黑色
                cv2.fillPoly(img, [points], color)

        # Step2: 画 Parking_slot
        for anno in data["anno"]:
            object_type = anno["category"]["type"]
            if object_type == "Parking_slot":
                all_points_x = anno["data"]["allPointsX"]
                all_points_y = anno["data"]["allPointsY"]
                points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                
                color = type_colors.get(object_type, (0, 0, 0))  # 默认黑色
                cv2.fillPoly(img, [points], color)

        # Step3: 画 ParkingLine
        for anno in data["anno"]:
            object_type = anno["category"]["type"]
            if object_type == "Parking_line":
                all_points_x = anno["data"]["allPointsX"]
                all_points_y = anno["data"]["allPointsY"]
                points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                
                color = type_colors.get(object_type, (0, 0, 0))  # 默认黑色
                cv2.fillPoly(img, [points], color)

        # Step4: 画 Center_lane
        for anno in data["anno"]:
            object_type = anno["category"]["type"]
            if object_type == "Center_lane":
                all_points_x = anno["data"]["allPointsX"]
                all_points_y = anno["data"]["allPointsY"]
                points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                
                color = type_colors.get(object_type, (0, 0, 0))  # 默认黑色
                cv2.fillPoly(img, [points], color)


        # Step5: 画其他类型
        for anno in data["anno"]:
            object_type = anno["category"]["type"]
            if object_type != "Road" and object_type != "Parking_slot" and object_type != "Parking_line" and object_type != "self_car" and object_type != "Center_lane":
                all_points_x = anno["data"]["allPointsX"]
                all_points_y = anno["data"]["allPointsY"]
                points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                
                color = type_colors.get(object_type, (0, 0, 0))  # 默认黑色
                cv2.fillPoly(img, [points], color)

        for anno in data["anno"]:
            object_type = anno["category"]["type"]
            if object_type == "self_car":
                all_points_x = anno["data"]["allPointsX"]
                all_points_y = anno["data"]["allPointsY"]
                points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                
                color = type_colors.get(object_type, (0, 0, 0))  # 默认黑色
                cv2.fillPoly(img, [points], color)
        # 保存图像
        # output_image_path = os.path.join(seg_dir, seg_filename)
        # cv2.imwrite(output_image_path, img)
        # print(f"已生成图像并保存: {output_image_path}")
        return img

    def replace_right_side_with_seg(self, original_image_path, seg_image):
        # 加载原始图像
        original_image = PIL.Image.open(original_image_path)
        # 将 BGR 转换为 RGB
        seg_image_rgb = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
        # 加载新的分割图
        seg_image = PIL.Image.fromarray(seg_image_rgb)
        # 确保分割图的大小为 896x896
        seg_image = seg_image.resize((896, 896))
        # 创建一个新的图像，大小与原始图像相同
        new_image = PIL.Image.new("RGB", original_image.size)
        # 将原始图像的左侧部分复制到新图像
        new_image.paste(original_image.crop((0, 0, original_image.width - 896, original_image.height)), (0, 0))
        # 将分割图粘贴到新图像的右侧
        new_image.paste(seg_image, (original_image.width - 896, 0))
        # 保存修改后的图像
        new_image.save(original_image_path)
        # print(f"已生成新avm图像并保存: {original_image_path}")
        logger.info(f"已生成新avm图像并保存: {original_image_path}")

    def replace_right_side_with_slot(self, original_image_path):
        try:
            # 加载原始图像
            original_image = PIL.Image.open(original_image_path)
            seg_image = np.asarray(original_image.crop((0, 0, original_image.width - 896, original_image.height)))
            if not seg_image.flags.writeable:
                seg_image = np.copy(seg_image)  # 创建可写副本
            seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
            # print(original_image, seg_image)
            
            # 创建一个新的图像，大小与原始图像相同
            new_image = PIL.Image.new("RGB", original_image.size)
            
            # 将原始图像的左侧部分复制到新图像
            new_image.paste(original_image.crop((0, 0, original_image.width - 896, original_image.height)), (0, 0))
            
            # 如果没有提供 seg_image，则创建一个新的空白图像
            if True:
                # 如果提供了 JSON 文件路径，则绘制标注
                if hasattr(self, 'filename') and self.filename:
                    # 读取 JSON 文件
                    with open(self.filename, "rb") as f:
                        data = orjson.loads(f.read())
                    
                    # 遍历 JSON 中的每个注释
                    for annotation in data['anno']:
                        child_type = annotation['category']['child']['type']
                        
                        if child_type == 'keypoint':
                            # 绘制关键点
                            x = int(annotation['data']['x'])
                            y = int(annotation['data']['y'])
                            cv2.circle(seg_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # 红色关键点
                        
                        elif child_type == 'line':
                            # 绘制普通线（蓝色）
                            all_points_x = annotation['data']['allPointsX']
                            all_points_y = annotation['data']['allPointsY']
                            points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                            cv2.polylines(seg_image, [points], False, color=(255, 255, 0), thickness=1)  # 蓝色线
                        
                        elif child_type == 'entrance_line':
                            # 绘制入口线（天蓝色）
                            all_points_x = annotation['data']['allPointsX']
                            all_points_y = annotation['data']['allPointsY']
                            points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                            cv2.polylines(seg_image, [points], False, color=(255, 0, 0), thickness=1)  # 天蓝色线
                        
                        elif child_type == 'rear_line':
                            # 绘制rear线（橙色）
                            all_points_x = annotation['data']['allPointsX']
                            all_points_y = annotation['data']['allPointsY']
                            points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
                            cv2.polylines(seg_image, [points], False, color=(0, 165, 255), thickness=1)  # 橙色线
            
            # 将 BGR 转换为 RGB
            seg_image_rgb = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
            
            # 加载新的分割图
            seg_image_pil = PIL.Image.fromarray(seg_image_rgb)
            
            # 确保分割图的大小为 896x896
            seg_image_pil = seg_image_pil.resize((896, 896))
            
            # 将分割图粘贴到新图像的右侧
            new_image.paste(seg_image_pil, (original_image.width - 896, 0))
            
            # 保存修改后的图像
            new_image.save(original_image_path)
            # print(f"已生成新avm图像并保存: {original_image_path}")
            logger.info(f"已生成新avm图像并保存: {original_image_path}")
            
        except Exception as e:
            print(f"替换右侧图像时出错: {e}")

    def GetCXColor(self, cxType):
        CXType2Color = {
            "standing_pedestrian": "#808708",
            "car": "#FFF200",
            "rider": "#68818C",
            "truck": "#FFFFFF",
            "bus": "#FFFFFF",
            "motorbike": "#75F305",
            "pillar": "#70E2AA",
            "anticollision": "#E03F23",
            "bumping_post": "#FFFFFF",
            "traffic_cones": "#5EF2F2",
            "shopping_cart": "#FFFFFF",
            "babycart": "#FFFFFF",
            "drum": "#99C4C3",
            "Turnstile_opened": "#FFFFFF",
            "Turnstile_closed": "#FFFFFF",
            "fire_hydrant": "#924F44",
            "electric_closet": "#FFFFFF",
            "charging_pile": "#FFFFFF",
            "ground_lock_open": "#488D7F",
            "ground_lock_close": "#BCE784",
            "limiter": "#827AF9",
            "limiter_pole": "#FFFFFF",
            "front": "#AC5AA1",
            "rear": "#429B1D",
            "light": "#191479",
            "plate" :"#A6261B"
        }
        try:
            cxColor = CXType2Color[cxType]
        except:
            cxColor = "#FFFFFF"
        return cxColor

    def hex_to_rgb(self, hex_color):
        # 去掉前面的 '#' 符号
        hex_color = hex_color.lstrip('#')
        
        # 将每两个字符转换为十进制数
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        return (b, g, r)

    
    def replace_right_side_with_2dod(self, original_image_path, new_path):
        try:
            # 加载原始图像
            original_image = PIL.Image.open(original_image_path)
            new_image = np.asarray(original_image)

            if not new_image.flags.writeable:
                new_image = np.copy(new_image)  # 创建可写副本
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

            # 如果提供了 JSON 文件路径，则绘制标注
            if hasattr(self, 'filename') and self.filename:
                # 读取 JSON 文件
                with open(self.filename, "rb") as f:
                    data = orjson.loads(f.read())

            # 在新图像上绘制 2DOD 标注
            for annotation in data['anno']:
                if True:  #shape["label"] == "2DOD":
                    object_type = annotation["category"]["type"]
                
                    # 提取 x, y, width, height
                    x = annotation["data"]["x"]
                    y = annotation["data"]["y"]
                    width = annotation["data"]["width"]
                    height = annotation["data"]["height"]

                    # 计算矩形的四个角点
                    points = [
                        (x, y),
                        (x + width, y),
                        (x + width, y + height),
                        (x, y + height)
                    ]

                    # 计算矩形的左上角和右下角坐标
                    x_min = min(point[0] for point in points)
                    y_min = min(point[1] for point in points)
                    x_max = max(point[0] for point in points)
                    y_max = max(point[1] for point in points)

                    # 获取颜色
                    color = self.GetCXColor(object_type)
                    rgb_color = self.hex_to_rgb(color)

                    # 绘制矩形框
                    # print(x_min, y_min)
                    cv2.rectangle(new_image, (int(round(x_min)), int(round(y_min))), (int(round(x_max)), int(round(y_max))), rgb_color, 1)  # 使用获取的颜色

                    # 在矩形框外绘制类型标签
                    label_position = (int(round(x_min)), int(round(y_min)) - 10)
                    # 计算标签位置
                    # label_x = int(round(x_min))
                    # label_y = int(round(y_min)) - 10  # 在矩形框上方绘制标签

                    # # 确保标签位置在画幅范围内
                    # label_x = max(100, min(label_x, 1100))  # 确保 x 坐标在 [0, 1280] 范围内
                    # label_y = max(0, min(label_y, 900))   # 确保 y 坐标在 [0, 960] 范围内
                    cv2.putText(new_image, object_type, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_color, 1, cv2.LINE_AA)

            # 保存修改后的图像
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            # 加载新的分割图
            new_image = PIL.Image.fromarray(new_image)
            new_image.save(new_path)
            logger.info(f"已生成新 2D-OD 图像并保存: {new_path}")

        except Exception as e:
            logger.error(f"绘制 2D-OD 图像时出错: {e}")

    def save_2dod(self, filename, shapes):
        try:
            data = {
                "anno": [],
                "index": "",
                "publicAttrs": {
                    "fileWidth": 1280,
                    "fileHeight": 960
                },
                "version": "7.0"
            }

            # 处理每个形状
            for i, shape in enumerate(shapes):
                shape_type = shape["shape_type"]
                label = shape["label"]
                points = shape["points"]
                # print(shape_type, label, points)

                if shape_type == "polygon":
                    # 计算矩形的 x, y, width, height
                    x = points[0][0]
                    y = points[0][1]
                    width = points[1][0] - x
                    height = points[2][1] - y

                    annotation = {
                        "attrs": {shape["description"].split("+")[0]: shape["description"].split("+")[1]},
                        "category": {
                            "child": {
                                "attributes": {},
                                "type": label
                            },
                            "type": label
                        },
                        "create": "MARK",
                        "data": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "name": "rect"
                        },
                        "fileMetaUuid": shape["description"].split("+")[2],
                        "id": shape["description"].split("+")[3],
                        "objectId": shape["description"].split("+")[4],
                        "shapeKey": "rect"
                    }
                    data["anno"].append(annotation)

                if shape_type == "rectangle":
                    # 计算矩形的 x, y, width, height
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    # 计算 x 和 y 的最小值
                    x = min(x1, x2)
                    y = min(y1, y2)
                    # 计算 width 和 height 的绝对值
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    objectId = str(random.randint(1000000, 9999999))

                    annotation = {
                        "attrs": {label: self.GetCXColor(label)},
                        "category": {
                            "child": {
                                "attributes": {},
                                "type": label
                            },
                            "type": label
                        },
                        "create": "MARK",
                        "data": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "name": "rect"
                        },
                        "fileMetaUuid": str(uuid.uuid4()),
                        "id": str(uuid.uuid4()),
                        "objectId": objectId,
                        "shapeKey": "rect"
                    }
                    data["anno"].append(annotation)

            # 保存到文件
            with open(filename, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
                logger.info(f"已生成新 2D-OD json 并保存: {filename}")

            self.filename = filename

            # 更新可视化图像
            parent_dir = osp.dirname(osp.abspath(filename))
            
            # 更新 vis_avm 图
            image_dir = osp.join(osp.dirname(parent_dir), "image")
            avm_dir = osp.join(osp.dirname(parent_dir), "vis_avm")
            original_image_path = osp.join(image_dir, osp.splitext(osp.basename(filename))[0] + ".jpg")
            new_path = osp.join(avm_dir, osp.splitext(osp.basename(filename))[0] + ".png")
            self.replace_right_side_with_2dod(original_image_path, new_path)

        except Exception as e:
            logger.error(e)
            raise LabelFileError(e)

    def save_slot(self, filename, shapes):
        try:
            # 创建一个新的数据结构，用于保存 Slot 标注
            data = {
                "anno": [],
                "index": "",
                "publicAttrs": {
                    "fileHeight": 896,
                    "fileWidth": 896
                },
                "version": "7.0"
            }
            
            # 处理每个形状
            for i, shape in enumerate(shapes):
                shape_type = shape["shape_type"]
                label = shape["label"]
                points = shape["points"]
                
                # 处理线类型
                if shape_type == "line":
                    annotation = {
                        "attrs": {
                            "line": "#FF0000"  # 可以根据需要设置颜色
                        },
                        "category": {
                            "child": {
                                "attributes": {},
                                "type": label
                            },
                            "type": label
                        },
                        "create": "MARK",
                        "data": {
                            "allPointsX": [point[0] for point in points],
                            "allPointsY": [point[1] for point in points],
                            "name": "line"
                        },
                        "fileMetaUuid": "0x35",
                        "id": i + 1,
                        "objectId": i + 1,
                        "preAnnotationId": i + 1,
                        "shapeKey": "line"
                    }
                    data["anno"].append(annotation)
                
                # 处理点类型
                elif shape_type == "point":
                    annotation = {
                        "attrs": {
                            "keypoint": "#FF0000"  # 可以根据需要设置颜色
                        },
                        "category": {
                            "child": {
                                "attributes": {},
                                "type": "keypoint"
                            },
                            "type": "keypoint"
                        },
                        "create": "MARK",
                        "data": {
                            "x": points[0][0],
                            "y": points[0][1]
                        },
                        "fileMetaUuid": "0x35",
                        "id": i + 1,
                        "objectId": i + 1,
                        "preAnnotationId": i + 1,
                        "shapeKey": "point"
                    }
                    data["anno"].append(annotation)
                
                # 处理自车类型
                elif label == "self_car":
                    annotation = {
                        "attrs": {
                            "self_vehicle": "#000000"  # 可以根据需要设置颜色
                        },
                        "category": {
                            "child": {
                                "attributes": {},
                                "type": "self_vehicle"
                            },
                            "type": "self_vehicle"
                        },
                        "create": "MARK",
                        "data": {
                            "allPointsX": [point[0] for point in points],
                            "allPointsY": [point[1] for point in points],
                            "name": "self_vehicle"
                        },
                        "fileMetaUuid": "a3a9f374-af6d-43d0-bd1a-dfb778ac013b",
                        "id": i + 1,
                        "objectId": "32aebd5d-25b8-455f-a5f4-4c191b79d393",  # 生成唯一ID
                        "preAnnotationId": str(i + 1),
                        "shapeKey": "line"
                    }
                    data["anno"].append(annotation)
            
            # 保存到文件
            with open(filename, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            
            self.filename = filename
            
            # 更新可视化图像
            parent_dir = osp.dirname(osp.abspath(filename))
            
            # 更新 vis_avm 图
            avm_dir = osp.join(osp.dirname(parent_dir), "vis_avm")
            original_image_path = osp.join(avm_dir, osp.splitext(osp.basename(filename))[0] + ".png")
            self.replace_right_side_with_slot(original_image_path)
            
        except Exception as e:
            raise LabelFileError(e)

    def save(self, filename, shapes):
        # 检查目录名称
        dir_name = osp.basename(osp.dirname(osp.dirname(filename)))
        if dir_name.startswith("Slot_"):
            return self.save_slot(filename, shapes)
        if dir_name.startswith("2D-OD_"):
            return self.save_2dod(filename, shapes)
        
        # 加载原始 JSON 文件
        with open(filename, "rb") as f:
            original_json = orjson.loads(f.read())

        # 过滤出非 Parking_slot 和 Parking_line 的形状
        original_data = original_json["anno"]
        original_data_with_ori_label = [
            annotation for annotation in original_data
            if annotation["category"]["type"] not in need_show_list
        ]
        data = {
            "anno": [
                {
                    "attrs": {shapes[num]["description"].split("+")[0]: shapes[num]["description"].split("+")[1]},
                    "category": {
                        "child": {
                            "attributes": {"Attribute": shapes[num]["description"].split("+")[2]},
                            "type": shapes[num]["label"],
                        },
                        "type": shapes[num]["label"],
                    },
                    "create": "0",  # 根据需要设置创建时间或状态
                    "data": {
                        "allPointsX": [point[0] for point in shapes[num]["points"]],
                        "allPointsY": [point[1] for point in shapes[num]["points"]],
                        "name": "polygon_close"
                    },
                    "fileMetaUuid": "0",
                    "id": num + 1,
                    "objectId": num + 1,
                    "preAnnotationId": num + 1,
                    "shapeKey": "polygon_close"
                }
                for num in range(len(shapes))
            ],
        }

        data["anno"].extend(original_data_with_ori_label)

        data["index"] = ""

        data["publicAttrs"] =  {
            "fileHeight": 896,
            "fileWidth": 896
        }

        data["version"] = "7.0"

        # with open(filename, "w") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            with open(filename, "wb") as f:  # 以二进制模式写入文件
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))  # 使用 orjson 写入数据
            self.filename = filename
            
        except Exception as e:
            raise LabelFileError(e)
        
        # 获取JSON文件的上一级文件夹
        parent_dir = osp.dirname(osp.abspath(filename))
        new_seg_img = self.draw_seg(filename)

        # 绘制新的 vis_avm 图
        avm_dir = osp.join(osp.dirname(parent_dir), "vis_avm")  # 上一级文件夹下的 vis_avm 文件夹
        original_image_path = osp.join(avm_dir, osp.splitext(osp.basename(filename))[0] + ".png")  # 构建新的JPG文件路径
        self.replace_right_side_with_seg(original_image_path, new_seg_img)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix