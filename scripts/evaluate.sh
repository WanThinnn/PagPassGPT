# File này đánh giá chất lượng mật khẩu sinh ra

dataset_name="rockyou"
test_dataset="./dataset/${dataset_name}-cleaned-Test.txt"  # Tập test
# output_path nên thay đổi số lượng sinh tương ứng với generate_num trong generate.sh
output_path="./generate/1000000/"  # Thư mục chứa mật khẩu sinh ra

# 1. Đánh giá mật khẩu sinh ra bằng phương pháp thông thường
python3.8 evaluate.py --test_file="$test_dataset" --gen_path="$gen_path" --isNormal

# 2. Đánh giá mật khẩu sinh ra bằng phương pháp DC-GEN
python3.8 evaluate.py --test_file="$test_dataset" --gen_path="$gen_path"