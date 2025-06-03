import os
import shutil

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs("train_images", exist_ok=True)
os.makedirs("test_images", exist_ok=True)
os.makedirs("train_labels", exist_ok=True)
os.makedirs("test_labels", exist_ok=True)

# Hàm di chuyển ảnh .jpg
def move_images(src_folder, dst_folder):
    for file in os.listdir(src_folder):
        if file.lower().endswith(".jpg"):
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(dst_folder, file)
            shutil.copy2(src_path, dst_path)

# Hàm di chuyển file .xml
def move_labels(src_folder, dst_folder):
    for file in os.listdir(src_folder):
        if file.lower().endswith(".xml"):
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(dst_folder, file)
            shutil.copy2(src_path, dst_path)

# Thực thi
move_images("train", "train_images")
move_images("test", "test_images")

move_labels("train", "train_labels")
move_labels("test", "test_labels")

print("Đã chia ảnh và nhãn vào các thư mục tương ứng: train_images/test_images & train_labels/test_labels.")
