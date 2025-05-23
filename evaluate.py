# This file aims to evaluate the passwords generated by different models (mainly on Hit Rate and repeat rate).
'''
File này được viết để đánh giá các mật khẩu được tạo ra bởi các mô hình khác nhau (chủ yếu là tỷ lệ trúng và tỷ lệ lặp lại).
Nó sẽ đọc các mật khẩu từ một tệp đầu vào, phân tích chúng thành các phần tử mô tả kiểu ký tự và sau đó tính toán tỷ lệ trúng và tỷ lệ lặp lại của các mật khẩu được tạo ra.

'''
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--test_file", help="file of test set", type=str, required=True) # đường dẫn đến tệp đầu vào chứa mật khẩu cần đánh giá
parser.add_argument("--gen_path", help="path of generated password", type=str, required=True) # đường dẫn đến thư mục chứa các tệp mật khẩu được tạo ra
parser.add_argument("--isNormal", action="store_true", help="whether the generated password is in normal method") # tham số boolean để xác định xem mật khẩu được tạo ra có phải theo phương pháp bình thường hay không
args = parser.parse_args()

if args.isNormal: # nếu mật khẩu được tạo ra theo phương pháp bình thường
    keyWord = "Normal"  # từ khóa để tìm kiếm trong tên tệp
else:
    keyWord = "DC" # từ khóa để tìm kiếm trong tên tệp

def get_all_files(path): 
    '''
    Hàm này nhận vào một đường dẫn và tìm tất cả các tệp trong thư mục đó và các thư mục con của nó.
    Nó sẽ trả về một danh sách chứa đường dẫn đầy đủ của tất cả các tệp có chứa từ khóa được chỉ định trong tên tệp.
    
    :param path: đường dẫn đến thư mục cần tìm kiếm
    :return: danh sách chứa đường dẫn đầy đủ của tất cả các tệp có chứa từ khóa trong tên tệp
    '''
    files = [] # danh sách để lưu trữ các tệp tìm thấy
    for root, dirs, filenames in os.walk(path): # lặp qua tất cả các thư mục và tệp trong đường dẫn
        for filename in filenames: # lặp qua tất cả các tệp trong thư mục hiện tại
            if keyWord in filename: # kiểm tra xem tên tệp có chứa từ khóa hay không
                files.append(os.path.join(root, filename)) # nếu có, thêm đường dẫn đầy đủ của tệp vào danh sách
    return files # trả về danh sách các tệp tìm thấy

gen_files = get_all_files(args.gen_path) # gọi hàm để tìm tất cả các tệp trong thư mục chứa mật khẩu được tạo ra


def get_gen_passwords(gen_files, isNormal):
    '''
    Hàm này nhận vào danh sách các tệp mật khẩu được tạo ra và một tham số boolean để xác định phương pháp tạo mật khẩu.
    Nó sẽ đọc từng tệp và phân tích các mật khẩu được tạo ra theo phương pháp bình thường hoặc phương pháp khác.
    Nó sẽ trả về một danh sách chứa tất cả các mật khẩu được tạo ra.
    
    :param gen_files: danh sách các tệp mật khẩu được tạo ra
    :param isNormal: tham số boolean để xác định phương pháp tạo mật khẩu
    :return: danh sách chứa tất cả các mật khẩu được tạo ra
    '''
    gen_passwords = [] # danh sách để lưu trữ các mật khẩu được tạo ra
    for gen_file in gen_files: # lặp qua tất cả các tệp mật khẩu được tạo ra
        if isNormal: # nếu mật khẩu được tạo ra theo phương pháp bình thường
            with open(gen_file, "r") as f: # mở tệp với chế độ đọc
                for line in f.readlines():  # lặp qua từng dòng trong tệp
                    try: # kiểm tra xem dòng có chứa từ khóa "Normal" hay không
                        gen_passwords.append(line.split(" ")[1]) # nếu có, thêm mật khẩu vào danh sách
                    except: # nếu không, bỏ qua dòng đó
                        continue 
        else: # nếu mật khẩu được tạo ra theo phương pháp khác
            with open(gen_file, "r") as f: # mở tệp với chế độ đọc
                gen_passwords += f.readlines() # thêm tất cả các dòng trong tệp vào danh sách mật khẩu được tạo ra
    return gen_passwords # trả về danh sách các mật khẩu được tạo ra

def get_hit_rate(test_file, gen_files):
    '''
    Hàm này nhận vào đường dẫn đến tệp mật khẩu cần đánh giá và danh sách các tệp mật khẩu được tạo ra.
    Nó sẽ đọc các mật khẩu từ tệp đầu vào và so sánh với các mật khẩu được tạo ra.
    Nó sẽ tính toán tỷ lệ trúng (hit rate) bằng cách chia số lượng mật khẩu trùng khớp cho tổng số mật khẩu trong tệp đầu vào.
    
    :param test_file: đường dẫn đến tệp mật khẩu cần đánh giá
    :param gen_files: danh sách các tệp mật khẩu được tạo ra
    :return: tỷ lệ trúng (hit rate)
    '''
    hit_num = 0 # biến để đếm số lượng mật khẩu trùng khớp
    gen_passwords = get_gen_passwords(gen_files, args.isNormal) # gọi hàm để lấy danh sách các mật khẩu được tạo ra
    gen_passwords = set(gen_passwords)  # chuyển đổi danh sách thành tập hợp để loại bỏ các mật khẩu trùng lặp

    with open(test_file, "r") as f: # mở tệp đầu vào với chế độ đọc
        test_passwords = f.readlines() # đọc tất cả các dòng trong tệp
    test_passwords = set(test_passwords) # chuyển đổi danh sách thành tập hợp để loại bỏ các mật khẩu trùng lặp

    for password in gen_passwords: # lặp qua tất cả các mật khẩu được tạo ra
        if password in test_passwords: # kiểm tra xem mật khẩu có nằm trong danh sách mật khẩu cần đánh giá hay không
            hit_num += 1 # nếu có, tăng biến đếm lên 1
    
    hit_rate = hit_num / len(test_passwords) # tính toán tỷ lệ trúng bằng cách chia số lượng mật khẩu trùng khớp cho tổng số mật khẩu trong tệp đầu vào
    return hit_rate # trả về tỷ lệ trúng


def get_repeat_rate(gen_files):
    '''
    Hàm này nhận vào danh sách các tệp mật khẩu được tạo ra.
    Nó sẽ đọc tất cả các mật khẩu từ các tệp và tính toán tỷ lệ lặp lại (repeat rate) bằng cách chia số lượng mật khẩu trùng lặp cho tổng số mật khẩu được tạo ra.
    
    :param gen_files: danh sách các tệp mật khẩu được tạo ra
    :return: tỷ lệ lặp lại (repeat rate)
    '''
    gen_passwords = get_gen_passwords(gen_files, args.isNormal) # gọi hàm để lấy danh sách các mật khẩu được tạo ra
    _gen_passwords = set(gen_passwords) # chuyển đổi danh sách thành tập hợp để loại bỏ các mật khẩu trùng lặp
    repeat_rate = 1 - len(_gen_passwords) / len(gen_passwords) # tính toán tỷ lệ lặp lại bằng cách chia số lượng mật khẩu trùng lặp cho tổng số mật khẩu được tạo ra 
    return repeat_rate

hit_rate = get_hit_rate(args.test_file, gen_files) # gọi hàm để tính toán tỷ lệ trúng
repeat_rate = get_repeat_rate(gen_files) # gọi hàm để tính toán tỷ lệ lặp lại
print("Hit Rate: ", hit_rate) # in ra tỷ lệ trúng
print("Repeat Rate: ", repeat_rate) # in ra tỷ lệ lặp lại
