import os
import json
import random
import shutil

def main():
    # Paths search
    kaggle_paths = [
        "/kaggle/input/datasets/nta212/ip102-for-object-detection",
        "/kaggle/input/nta212/ip102-for-object-detection",
        "/kaggle/input/ip102-for-object-detection"
    ]
    local_input = "./datasets/IP102"
    
    input_dir = None
    for p in kaggle_paths:
        if os.path.exists(p):
            input_dir = p
            break
            
    if not input_dir:
        input_dir = local_input
        
    print(f"Using dataset source directory: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"Error: Dataset directory {input_dir} not found.")
        return
        
    train_json_path = os.path.join(input_dir, "train.json")
    test_json_path = os.path.join(input_dir, "test.json")
    val_json_path = os.path.join(input_dir, "val.json")
    if not os.path.exists(val_json_path):
        val_json_path = test_json_path # fallback to test if val is not present
        
    # 1. Read class names
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    categories = sorted(train_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    print(f"Loaded {len(class_names)} classes.")
    
    # 2. Setup symlinks / directories under datasets/
    os.makedirs("./datasets", exist_ok=True)
    images_src = os.path.join(input_dir, "VOC2007", "VOC2007", "JPEGImages")
    annos_src = os.path.join(input_dir, "VOC2007", "VOC2007", "Annotations")
    
    images_dest = "./datasets/JPEGImages"
    annos_dest = "./datasets/Annotations"
    
    # Remove existing symlinks or dirs if any
    for dest in [images_dest, annos_dest]:
        if os.path.exists(dest) or os.path.islink(dest):
            if os.path.islink(dest):
                os.unlink(dest)
            elif os.path.isdir(dest):
                shutil.rmtree(dest)
            else:
                os.remove(dest)
            
    # Create symlinks
    try:
        os.symlink(images_src, images_dest)
        os.symlink(annos_src, annos_dest)
        print("Created symlinks for JPEGImages and Annotations.")
    except Exception as e:
        print(f"Symlink failed: {e}. Trying to copy directories instead...")
        # Local fallback if symlink fails (e.g. on Windows without admin rights)
        if os.path.exists(images_src):
            shutil.copytree(images_src, images_dest)
            shutil.copytree(annos_src, annos_dest)
            print("Successfully copied JPEGImages and Annotations directories.")
        else:
            print("Source directories do not exist. Skipping symlink creation (expect this on local dev if not fully populated).")
        
    # 3. Create Main directory for task txt splits
    main_txt_dir = "./datasets/ImageSets/Main/IP102"
    os.makedirs(main_txt_dir, exist_ok=True)
    
    # Extract image list per class to build splits
    train_images = {img['id']: os.path.splitext(img['file_name'])[0] for img in train_data['images']}
    
    # Map class index to list of image names containing it
    class_to_images = {i: [] for i in range(25)}
    
    # Categories list mapping from original category ID to 0-24 index
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    
    for ann in train_data['annotations']:
        cat_id = ann['category_id']
        if cat_id in cat_id_to_idx:
            cls_idx = cat_id_to_idx[cat_id]
            img_id = ann['image_id']
            if img_id in train_images:
                img_name = train_images[img_id]
                class_to_images[cls_idx].append(img_name)
                
    # Define tasks
    task_classes = {
        1: list(range(0, 7)),     # 7 classes
        2: list(range(7, 13)),    # 6 classes
        3: list(range(13, 19)),   # 6 classes
        4: list(range(19, 25))    # 6 classes
    }
    
    # Write training image lists for each task
    for task_id, classes in task_classes.items():
        task_imgs = set()
        for cls in classes:
            task_imgs.update(class_to_images[cls])
        
        # Write to txt
        with open(os.path.join(main_txt_dir, f"t{task_id}.txt"), 'w') as f:
            for img in sorted(task_imgs):
                f.write(img + '\n')
        print(f"Task {task_id} training set: {len(task_imgs)} images.")
        
    # Write exemplar lists (fine-tuning splits) using K=20
    K = 20
    random.seed(42)
    
    for task_id in [2, 3, 4]:
        # Classes seen so far: all classes in tasks up to current task
        current_seen_classes = []
        for t in range(1, task_id + 1):
            current_seen_classes.extend(task_classes[t])
            
        ft_imgs = set()
        for cls in current_seen_classes:
            cls_imgs = class_to_images[cls]
            if len(cls_imgs) > 0:
                selected = random.sample(cls_imgs, min(K, len(cls_imgs)))
                ft_imgs.update(selected)
                
        # Write to txt
        with open(os.path.join(main_txt_dir, f"t{task_id}_ft.txt"), 'w') as f:
            for img in sorted(ft_imgs):
                f.write(img + '\n')
        print(f"Task {task_id} FT set (exemplars): {len(ft_imgs)} images.")
        
    # Write test.txt and val.txt from test.json
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    test_imgs = [os.path.splitext(img['file_name'])[0] for img in test_data['images']]
    
    with open(os.path.join(main_txt_dir, "test.txt"), 'w') as f:
        for img in sorted(test_imgs):
            f.write(img + '\n')
    print(f"Test set: {len(test_imgs)} images.")
    
    # val.txt is same as test.txt
    shutil.copy(os.path.join(main_txt_dir, "test.txt"), os.path.join(main_txt_dir, "val.txt"))
    print("Val set (same as test set): generated.")
    
    # 4. Generate configs/IP102/ base and task yamls
    os.makedirs("./configs/IP102", exist_ok=True)
    
    sowodb_weight = "/kaggle/input/models/chienkhu/orthogonaldet-on-sowodb/pytorch/default/1/ours_s4_463.pth"
    if os.path.exists(sowodb_weight):
        base_weight = sowodb_weight
        print(f"Using S-OWODB pre-trained weight: {base_weight}")
    else:
        base_weight = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
        print(f"Using default COCO pre-trained weight: {base_weight}")
    
    # base.yaml
    base_content = f"""MODEL:
  META_ARCHITECTURE: "RandBox"
  WEIGHTS: "{base_weight}"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  BACKBONE:
    NAME: "build_resnet_fpn_backbone"
  RESNETS:
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]
  FPN:
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
  ROI_HEADS:
    IN_FEATURES: ["p2", "p3", "p4", "p5"]
  ROI_BOX_HEAD:
    POOLER_TYPE: "ROIAlignV2"
    POOLER_RESOLUTION: 7
    POOLER_SAMPLING_RATIO: 2
SOLVER:
  IMS_PER_BATCH: 12
  BASE_LR: 0.000025
  STEPS: (60000, 80000)
  MAX_ITER: 90000
  WARMUP_FACTOR: 0.01
  WARMUP_ITERS: 1000
  WEIGHT_DECAY: 0.0001
  OPTIMIZER: "ADAMW"
  BACKBONE_MULTIPLIER: 1.0
  CLIP_GRADIENTS:
    ENABLED: True
    CLIP_TYPE: "full_model"
    CLIP_VALUE: 1.0
    NORM_TYPE: 2.0
SEED: 40244023
INPUT:
  MIN_SIZE_TRAIN: (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
  CROP:
    ENABLED: False
    TYPE: "absolute_range"
    SIZE: (384, 600)
  FORMAT: "RGB"
TEST:
  EVAL_PERIOD: 10000
  MASK: 1
  SCORE_THRESH: 0.0
DATALOADER:
  FILTER_EMPTY_ANNOTATIONS: False
  NUM_WORKERS: 4
VERSION: 2
OUTPUT_DIR: "output/IP102/"
"""
    with open("./configs/IP102/base.yaml", 'w') as f:
        f.write(base_content)
        
    # t1.yaml
    t1_content = f"""_BASE_: "base.yaml"
MODEL:
  WEIGHTS: "{base_weight}"
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
  NUM_PROPOSALS: 500
  NUM_CLASSES: 26
DATASETS:
  TRAIN: ("my_train",)
  TEST:  ("my_val",)
SOLVER:
  STEPS: (15000, 18000)
  MAX_ITER: 20000
TEST:
  EVAL_PERIOD: 20000
  PREV_INTRODUCED_CLS: 0
  CUR_INTRODUCED_CLS: 7
INPUT:
  CROP:
    ENABLED: True
  FORMAT: "RGB"
"""
    with open("./configs/IP102/t1.yaml", 'w') as f:
        f.write(t1_content)
        
    # task config templates helper
    def write_task_yaml(task_name, prev_intro, cur_intro, prev_classes, mask):
        content = f"""_BASE_: "base.yaml"
MODEL:
  WEIGHTS: "{base_weight}"
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
  NUM_PROPOSALS: 500
  NUM_CLASSES: 26
DATASETS:
  TRAIN: ("my_train",)
  TEST:  ("my_val",)
SOLVER:
  STEPS: (30000,)
  MAX_ITER: 35000
TEST:
  EVAL_PERIOD: 35000
  PREV_INTRODUCED_CLS: {prev_intro}
  CUR_INTRODUCED_CLS: {cur_intro}
  PREV_CLASSES: {prev_classes}
  MASK: {mask}
INPUT:
  CROP:
    ENABLED: True
  FORMAT: "RGB"
"""
        with open(f"./configs/IP102/{task_name}.yaml", 'w') as f:
            f.write(content)
            
    # Task 2
    write_task_yaml("t2", prev_intro=7, cur_intro=6, prev_classes="(7,)", mask=2)
    write_task_yaml("t2_ft", prev_intro=7, cur_intro=6, prev_classes="(7,)", mask=1)
    # Task 3
    write_task_yaml("t3", prev_intro=13, cur_intro=6, prev_classes="(7, 6,)", mask=2)
    write_task_yaml("t3_ft", prev_intro=13, cur_intro=6, prev_classes="(7, 6,)", mask=1)
    # Task 4
    write_task_yaml("t4", prev_intro=19, cur_intro=6, prev_classes="(7, 6, 6,)", mask=2)
    write_task_yaml("t4_ft", prev_intro=19, cur_intro=6, prev_classes="(7, 6, 6,)", mask=1)
    
    print("Successfully generated all configurations under configs/IP102/")

if __name__ == '__main__':
    main()
