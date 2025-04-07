"""
This file aims to split whole dataset into training set and test set:
    
    Training set will be used in training and will be split again (for validation).
    Test set will be used in evaluation.

"""
"""
File này nhằm mục đích chia tập dữ liệu thành tập huấn luyện và tập kiểm tra:
    Tập huấn luyện sẽ được sử dụng trong quá trình huấn luyện và sẽ được chia lại (để xác thực).
    Tập kiểm tra sẽ được sử dụng trong quá trình đánh giá.
"""


import random 
import argparse

def split_train_test(file_path, ratio, train_path, test_path):
    """
    Hàm này dùng để chia tập dữ liệu thành 2 phần: train và test.
    :param file_path: đường dẫn đến tập dữ liệu gốc 
    :param ratio: tỷ lệ chia giữa train và test (mặc định là 0.8)
    :param train_path: đường dẫn để lưu tập train sau khi chia
    :param test_path: đường dẫn để lưu tập test sau khi chia
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print('Shuffling passwords.') # shuffle the lines
    # Trộn ngẫu nhiên các dòng trong tập dữ liệu
    random.shuffle(lines)
    f.close()

    split = int(len(lines) * ratio) # tính toán số lượng dòng cho tập train
    # Tính toán số lượng dòng cho tập train dựa trên tỷ lệ đã cho

    with open(train_path, 'w') as f:
        print('Saving 80% ({}) of dataset for training in {}'.format(split, train_path)) # save train set
        # Lưu tập train vào đường dẫn đã cho
        f.write(''.join(lines[0:split]))
    f.close()

    with open(test_path, 'w') as f:
        print('Saving 20% ({}) of dataset for test in {}'.format(len(lines) - split, test_path)) # save test set
        # Lưu tập test vào đường dẫn đã cho
        f.write(''.join(lines[split:]))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="path of cleaned dataset", type=str, required=True)
    parser.add_argument("--train_path", help="save path of training set after split", type=str, required=True)
    parser.add_argument("--test_path", help="save path of test set after split", type=str, required=True)
    parser.add_argument("--ratio", help="split ratio", type=float, default=0.8)
    args = parser.parse_args()

    print(f'Split begin.')
    split_train_test(args.dataset_path, args.ratio, args.train_path, args.test_path) # Chia tập dữ liệu thành 2 phần: train và test
    print(f'Split done.')