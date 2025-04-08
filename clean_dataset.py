"""
This file aims to implement data cleaning:

    We retained passwords with lengths ranging between 4 and 12 characters.
    We removed duplicate passwords.
    We removed the passwords containing Non-ASCII characters and invisible ASCII characters.

"""

'''
File này được viết để thực hiện việc làm sạch dữ liệu:
    Nó giữ lại các mật khẩu có độ dài từ 4 đến 12 ký tự.
    Nó loại bỏ các mật khẩu trùng lặp.
    Nó loại bỏ các mật khẩu chứa ký tự không phải ASCII và ký tự ASCII vô hình.
    Nó sử dụng thư viện argparse để xử lý các tham số dòng lệnh và thư viện os để làm việc với hệ thống tệp.

'''


import argparse

# Kiểm tra một mật khẩu có hợp lệ hay không
def filter_password(password):
    '''
    Hàm này nhận vào một mật khẩu và kiểm tra xem nó có hợp lệ hay không.
    Một mật khẩu được coi là hợp lệ nếu nó có độ dài từ 4 đến 12 ký tự và mỗi ký tự trong mật khẩu có mã ASCII nằm trong khoảng từ 33 đến 126 (không bao gồm các ký tự không in được).
    
    :param password: mật khẩu cần kiểm tra
    :return: True nếu mật khẩu hợp lệ, False nếu không hợp lệ
    '''
    # Độ dài mật khẩu phải từ 4 đến 12 ký tự
    if len(password) < 4 or len(password) > 12:
        return False
    # Mỗi ký tự trong mật khẩu phải có 32 < ASCII ≤ 126
    for ch in password:
        if ord(ch) > 126 or ord(ch) <= 32:
            return False
    return True

# Xử lý toàn bộ file dataset
def preprocess(password_path, output_path):
    # Mở file input (đọc) và file output (ghi)
    f = open(password_path, 'r', encoding='utf-8', errors='ignore')
    f_out = open(output_path, 'w', encoding='utf-8', errors='ignore')

    # Đọc tất cả các dòng từ file input
    lines = f.readlines()
    # Sử dụng set() để tự động loại bỏ các dòng trùng lặp
    lines = set(lines)
    total_num = 0
    valid_num = 0
    # Duyệt qua từng dòng trong file input, sau đó kiểm tra xem mật khẩu có hợp lệ hay không
    for line in lines:
        if not line:
            continue
        else:
            total_num += 1
            if filter_password(line[:-1]):
                valid_num += 1
                f_out.write(line)
    print('Total num={}'.format(total_num))
    print('Retain num={}'.format(valid_num))
    print('Retain rate:{}'.format(valid_num/total_num))
    f.close()
    f_out.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="path of original dataset", type=str, required=True)
    parser.add_argument("--output_path", help="path of cleaned dataset", type=str, required=True)
    args = parser.parse_args()

    print(f'Clean dataset begin.')
    preprocess(args.dataset_path, args.output_path)
    print(f'Clean dataset done.')