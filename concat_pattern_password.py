"""
This file aims to process the input passwords to the rule as follows for convience to train:
    
    pattern <SEP> password

"""

"""
File này được viết để xử lý các mật khẩu đầu vào theo quy tắc sau cho thuận tiện cho việc huấn luyện:
    pattern <SEP> password
"""


import argparse


def get_pattern(password:str):
    """
    Hàm này nhận vào một mật khẩu và trả về một danh sách các chuỗi mô tả kiểu ký tự của mật khẩu.
    Hàm này phân tích mật khẩu thành các phần tử dựa trên loại ký tự (chữ cái, số hoặc ký tự đặc biệt) và độ dài của mỗi phần tử.
    Mỗi phần tử trong danh sách được định dạng dưới dạng "L" cho chữ cái, "N" cho số và "S" cho ký tự đặc biệt, theo sau là độ dài của phần tử đó.
    Ví dụ: nếu mật khẩu là "abc123!@#", hàm sẽ trả về ['L3', 'N3', 'S4'].
    
    """
    result = []
    
    current_type = None
    current_length = 0
    
    for char in password:
        if char.isalpha():
            if current_type == 'L':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'L'
                current_length = 1
        elif char.isdigit():
            if current_type == 'N':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'N'
                current_length = 1
        else:
            if current_type == 'S':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'S'
                current_length = 1
    
    if current_type:
        result.append(current_type + str(current_length))
    return result


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="path of training dataset after split", type=str, required=True)
    parser.add_argument("--output_path", help="path of output dataset (ready for training)", type=str, required=True)
    args = parser.parse_args()
    
    input_dataset = args.dataset_path
    output_dataset = args.output_path
    f_in = open(input_dataset, 'r', encoding='utf-8', errors='ignore')
    f_out = open(output_dataset, 'w', encoding='utf-8', errors='ignore')

    lines = f_in.readlines()

    for line in lines:
        password = line[:-1]
        prompt = ' '.join(get_pattern(password))
        new_line = prompt + ' <SEP> ' + ' '.join(list(password)) + '\n'
        f_out.write(new_line)
