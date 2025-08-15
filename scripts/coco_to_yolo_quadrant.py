import json
import os
from PIL import Image

def convert_coco_to_yolo():
    coco_json_path = os.path.join(os.path.dirname(__file__), 'train_quadrant.json')
    images_dir = os.path.join(os.path.dirname(__file__), 'xrays')
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolo'))
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 只处理标准COCO类别字段
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_ids = list(categories.keys())
    category_ids.sort()
    with open(os.path.join(output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        for cat_id in category_ids:
            f.write(f"{categories[cat_id]}\n")

    images_info = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    for image_id, image_info in images_info.items():
        img_width = image_info['width']
        img_height = image_info['height']
        image_filename = image_info['file_name']
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        yolo_annotations = []
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                bbox = ann['bbox']
                x, y, w, h = bbox
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                category_id = ann['category_id']
                class_index = category_ids.index(category_id)
                yolo_annotations.append(f"{class_index} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    # 生成dataset.yaml
    class_names = [categories[cat_id] for cat_id in category_ids]
    yaml_content = f"""# YOLO数据集配置文件\npath: {output_dir}\ntrain: images/train\nval: images/val\nnc: {len(class_names)}\nnames: {class_names}\n"""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print('转换完成，YOLO标签保存在:', labels_dir)
    print('dataset.yaml已生成:', yaml_path)

if __name__ == "__main__":
    convert_coco_to_yolo()
