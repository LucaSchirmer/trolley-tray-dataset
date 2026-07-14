import os
from PIL import Image

def transform_yolo_coordinates(lines, transform_type):
    """
    Transforms normalized YOLO polygon or bounding box coordinates.
    Lines format: class_id x1 y1 x2 y2 ...
    """
    new_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        class_id = parts[0]
        # Group the remaining elements into (x, y) pairs
        coords = [float(x) for x in parts[1:]]
        new_coords = []
        
        for i in range(0, len(coords), 2):
            x = coords[i]
            y = coords[i+1]
            
            if transform_type == 'flip_h':
                # Horizontal flip: x becomes 1 - x, y stays same
                x = 1.0 - x
            elif transform_type == 'flip_v':
                # Vertical flip: x stays same, y becomes 1 - y
                y = 1.0 - y
            elif transform_type == 'rotate_90':
                # 90 deg clockwise: new_x = 1 - y, new_y = x
                orig_x = x
                x = 1.0 - y
                y = orig_x
            elif transform_type == 'rotate_180':
                # 180 deg: x = 1 - x, y = 1 - y
                x = 1.0 - x
                y = 1.0 - y
            elif transform_type == 'rotate_270':
                # 270 deg clockwise: new_x = y, new_y = 1 - x
                orig_x = x
                x = y
                y = 1.0 - orig_x
                
            new_coords.extend([x, y])
            
        # Reconstruct the YOLO line string
        coord_str = " ".join([f"{c:.6f}" for c in new_coords])
        new_lines.append(f"{class_id} {coord_str}\n")
        
    return new_lines

def augment_yolo_dataset(directory_path):
    # Find all JPEGs
    valid_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
    files = os.listdir(directory_path)
    image_files = [f for f in files if f.endswith(valid_extensions)]
    
    # Avoid picking up already augmented images if rerun
    suffixes = ['_flip_h', '_flip_v', '_rotate_90', '_rotate_180', '_rotate_270']
    image_files = [f for f in image_files if not any(s in f for s in suffixes)]

    print(f"Found {len(image_files)} base images. Processing labels...")

    augmentations = [
        ('_flip_h', Image.FLIP_LEFT_RIGHT, 'flip_h'),
        ('_flip_v', Image.FLIP_TOP_BOTTOM, 'flip_v'),
        ('_rotate_90', Image.ROTATE_270, 'rotate_90'), # PIL ROTATE_270 is 90 deg clockwise
        ('_rotate_180', Image.ROTATE_180, 'rotate_180'),
        ('_rotate_270', Image.ROTATE_90, 'rotate_270')  # PIL ROTATE_90 is 270 deg clockwise
    ]

    for file_name in image_files:
        name_part, ext_part = os.path.splitext(file_name)
        img_path = os.path.join(directory_path, file_name)
        txt_path = os.path.join(directory_path, f"{name_part}.txt")
        
        # Check if corresponding YOLO text label exists
        if not os.path.exists(txt_path):
            print(f"Warning: No matching label file for {file_name}. Skipping.")
            continue
            
        try:
            # Read original coordinates
            with open(txt_path, 'r') as f:
                original_lines = f.readlines()
                
            with Image.open(img_path) as img:
                for suffix, pil_transform, transform_key in augmentations:
                    # 1. Transform and save the image
                    aug_img = img.transpose(pil_transform)
                    aug_img.save(os.path.join(directory_path, f"{name_part}{suffix}{ext_part}"))
                    
                    # 2. Transform and save the text coordinates
                    aug_lines = transform_yolo_coordinates(original_lines, transform_key)
                    with open(os.path.join(directory_path, f"{name_part}{suffix}.txt"), 'w') as f_out:
                        f_out.writelines(aug_lines)
                        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("YOLO dataset augmentation completed successfully!")

# To run the script, put your images and text files in the same folder and call:
augment_yolo_dataset('./delete_me_do_not_push')