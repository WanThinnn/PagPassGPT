# File này thực hiện việc sinh mật khẩu mới.

dataset_name="rockyou"
cleaned_dataset="./dataset/${dataset_name}-cleaned.txt"  # Dataset đã làm sạch
model_path="./model/last-step/"  # Đường dẫn đến mô hình đã huấn luyện
output_path="./generate/"  # Thư mục đầu ra cho mật khẩu sinh ra

# 1. Tính toán tỷ lệ pattern (cần cho DC-GEN)
python3.8 get_pattern_rate.py --dataset_path=$cleaned_dataset

# 2. Sinh mật khẩu sử dụng DC-GEN (phương pháp nâng cao)
# python3.8 DC-GEN.py --model_path=$model_path --output_path=$output_path --generate_num=1000000 --batch_size=5000 --gpu_num=2 --gpu_index=4

# 3. Hoặc sinh mật khẩu không sử dụng DC-GEN (phương pháp thông thường)
#python normal-gen.py --model_path=$model_path --output_path=$output_path --generate_num=1000000 --batch_size=5000 --gpu_num=2 --gpu_index=4
python3.8 normal-gen.py --model_path=$model_path --output_path=$output_path --generate_num=1000000 --batch_size=5000 --gpu_num=1 --gpu_index=0