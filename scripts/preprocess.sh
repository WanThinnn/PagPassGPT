# File này thực hiện tiền xử lý dữ liệu trước khi huấn luyện mô hình
dataset_name="rockyou"  # Tên dataset
original_dataset="./dataset/${dataset_name}.txt"  # Đường dẫn dataset gốc
cleaned_dataset="./dataset/${dataset_name}-cleaned.txt"  # Đường dẫn dataset đã làm sạch
training_dataset="./dataset/${dataset_name}-cleaned-Train.txt"  # Đường dẫn tập train
test_dataset="./dataset/${dataset_name}-cleaned-Test.txt"  # Đường dẫn tập test
ready4train_dataset="./dataset/${dataset_name}-cleaned-Train-ready.txt"  # Đường dẫn dataset sẵn sàng để train

# 1. Làm sạch dataset gốc
python3.8 clean_dataset.py --dataset_path=$original_dataset --output_path=$cleaned_dataset

# 2. Chia dataset thành tập train và test
python3.8 split_dataset.py --dataset_path=$cleaned_dataset --train_path=$training_dataset --test_path=$test_dataset

# 3. Kết hợp pattern và password để chuẩn bị cho training
python3.8 concat_pattern_password.py --dataset_path=$training_dataset --output_path=$ready4train_dataset