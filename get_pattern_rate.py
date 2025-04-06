# This file aims to get patterns rate from training set (cleaned).

from concat_pattern_password import get_pattern  # Nhập hàm get_pattern từ module concat_pattern_password
import argparse  # Thư viện để xử lý tham số dòng lệnh
import os  # Thư viện để làm việc với hệ thống tệp

# Tạo parser để xử lý tham số dòng lệnh
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="path of training dataset", type=str, required=True)  # Đường dẫn đến tập dữ liệu huấn luyện, bắt buộc nhập
parser.add_argument("--output_path", help="save path of pattern rate", type=str, default="patterns.txt")  # Đường dẫn lưu kết quả tỷ lệ pattern, mặc định là "patterns.txt"
args = parser.parse_args()  # Lưu các tham số đã được phân tích

train_dataset_path = args.dataset_path  # Gán đường dẫn tập dữ liệu huấn luyện từ tham số
PCFG_rate_file = args.output_path  # Gán đường dẫn file đầu ra từ tham số

# Kiểm tra nếu file đầu ra chưa tồn tại thì thực hiện tính toán
if not os.path.isfile(PCFG_rate_file):
    # Mở file tập dữ liệu để đọc với mã hóa UTF-8, bỏ qua lỗi nếu có
    f_in = open(train_dataset_path, 'r', encoding='utf-8', errors='ignore')
    # Mở file đầu ra để ghi với mã hóa UTF-8, bỏ qua lỗi nếu có
    f_out = open(PCFG_rate_file, 'w', encoding='utf-8', errors='ignore')

    pcfg_patterns_dict = {}  # Tạo dictionary để lưu pattern và số lần xuất hiện

    lines = f_in.readlines()  # Đọc tất cả các dòng từ file tập dữ liệu
    total_num = len(lines)  # Tính tổng số dòng (tổng số mật khẩu)
    # Duyệt qua từng dòng trong tập dữ liệu
    for line in lines:
        if not line:  # Bỏ qua nếu dòng rỗng
            continue
        password = line[:-1]  # Lấy mật khẩu, bỏ ký tự xuống dòng cuối cùng
        pcfg_pattern = ' '.join(get_pattern(password))  # Tạo pattern từ mật khẩu bằng hàm get_pattern, nối bằng khoảng trắng
        if pcfg_pattern in pcfg_patterns_dict:  # Nếu pattern đã tồn tại trong dictionary
            pcfg_patterns_dict[pcfg_pattern] += 1  # Tăng số đếm lên 1
        else:  # Nếu pattern chưa tồn tại
            pcfg_patterns_dict[pcfg_pattern] = 1  # Khởi tạo với số đếm là 1

    # Sắp xếp dictionary theo số lần xuất hiện (value) giảm dần
    pcfg_patterns_dict = dict(sorted(pcfg_patterns_dict.items(), key=lambda x: x[1], reverse=True))

    # Ghi kết quả vào file đầu ra
    for key, value in pcfg_patterns_dict.items():
        # Ghi pattern và tỷ lệ xuất hiện (value/total_num) vào file, cách nhau bằng tab
        f_out.write(f'{key}\t{value/total_num}\n')
