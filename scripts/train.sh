# File này thực hiện quá trình huấn luyện mô hình.

dataset_name="rockyou"
ready4train_dataset="./dataset/${dataset_name}-cleaned-Train-ready.txt"

# 1. Tạo file từ vựng (vocab)
python3.8 generate_vocab_file.py

# 2. Huấn luyện mô hình
python3.8 train.py --dataset_path=$ready4train_dataset