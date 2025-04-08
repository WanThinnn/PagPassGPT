'''
File này được viết để tạo ra một tệp vocab chứa các token và ID tương ứng cho các ký tự và pattern trong mật khẩu.
Nó sẽ tạo ra một tệp vocab.json chứa các token và ID tương ứng cho các ký tự và pattern trong mật khẩu.
Nó sẽ sử dụng thư viện json để lưu trữ dữ liệu dưới định dạng JSON và thư viện OrderedDict để giữ nguyên thứ tự của các phần tử trong dictionary.
Nó sẽ sử dụng thư viện argparse để xử lý các tham số dòng lệnh và thư viện os để làm việc với hệ thống tệp.
'''

import json  # Thư viện để làm việc với định dạng JSON
from collections import OrderedDict  # Nhập OrderedDict để tạo dictionary giữ nguyên thứ tự chèn
import argparse  # Thư viện để xử lý tham số dòng lệnh
import os  # Thư viện để làm việc với hệ thống tệp

# Tạo parser để xử lý tham số dòng lệnh
parser = argparse.ArgumentParser()
parser.add_argument("--save_path", help="save path of vocab file", type=str, default='./tokenizer/vocab.json')  # Đường dẫn lưu file vocab
parser.add_argument("--max_len", help="max length of password in datasets", default=12, type=int)  # Độ dài tối đa của mật khẩu
args = parser.parse_args()  # Lưu các tham số đã được phân tích

# Kiểm tra nếu file tại save_path chưa tồn tại thì thực hiện tạo vocab
if not os.path.isfile(args.save_path):
    # Định nghĩa các phạm vi mã ASCII
    Number = [(48, 57)]  # Số từ 0-9 (mã ASCII 48-57)
    Letter = [(65, 90), (97, 122)]  # Chữ cái in hoa (65-90) và thường (97-122)
    Special_char = [(33, 47), (58, 64), (91, 96), (123, 126)]  # Ký tự đặc biệt trong các khoảng ASCII

    chars = [Number, Letter, Special_char]  # Danh sách chứa các loại ký tự
    Special_token = ["<BOS>", "<SEP>", "<EOS>", "<UNK>", "<PAD>"]  # Các token đặc biệt: Bắt đầu, Phân cách, Kết thúc, Không xác định, Đệm

    vocab_dict = OrderedDict()  # Tạo dictionary có thứ tự để lưu vocab
    index = 0  # Biến đếm để gán ID cho mỗi token

    # Thêm các token đặc biệt vào vocab với ID tăng dần
    for token in Special_token:
        vocab_dict[token] = index  # Gán token với index hiện tại
        index += 1  # Tăng index cho token tiếp theo

    # Tạo các pattern token theo độ dài (N: Number, L: Letter, S: Special)
    for char_type in ['N', 'L', 'S']:
        for length in range(args.max_len, 0, -1):  # Từ max_len giảm dần về 1
            vocab_dict[char_type+str(length)] = index  # Thêm pattern như N12, L12, S12,...
            index += 1  # Tăng index

    # Thêm từng ký tự riêng lẻ từ các phạm vi ASCII
    for char_type in chars:
        for turple in char_type:  # Duyệt qua từng khoảng ASCII trong loại ký tự
            for i in range(turple[0], turple[1]+1):  # Duyệt từ giá trị bắt đầu đến kết thúc của khoảng
                vocab_dict[chr(i)] = index  # Chuyển mã ASCII thành ký tự và gán index
                index += 1  # Tăng index

    # Chuyển dictionary thành chuỗi JSON và lưu vào file
    json_str = json.dumps(vocab_dict, indent=4)  # Chuyển thành JSON với thụt lề 4 khoảng cách
    with open(args.save_path, 'w') as json_file:  # Mở file để ghi
        json_file.write(json_str)  # Ghi chuỗi JSON vào file
