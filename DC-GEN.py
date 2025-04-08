# This file aims to realize D&C-GEN.
"""
File này được viết để thực hiện D&C-GEN.
Nó sử dụng mô hình GPT-2 để sinh ra mật khẩu dựa trên các mẫu đã cho.

"""

from typing import Any
import torch
from transformers import GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList, LogitsProcessorList
from tokenizer import CharTokenizer
import time
import threading
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="directory of pagpassgpt", type=str, required=True) # đường dẫn đến mô hình GPT-2 đã được huấn luyện
parser.add_argument("--vocabfile_path", help="path of vocab file", type=str, default='./tokenizer/vocab.json') # đường dẫn đến tệp vocab.json
parser.add_argument("--pattern_path", help="path of pattern rate file", type=str, default='patterns.txt') # đường dẫn đến tệp chứa các mẫu mật khẩu và tỷ lệ của chúng
parser.add_argument("--output_path", help="directory of output file path", type=str, required=True) # đường dẫn đến thư mục đầu ra để lưu trữ mật khẩu đã sinh
parser.add_argument("--generate_num", help="total guessing number", default=1000000, type=int) # số lượng mật khẩu cần sinh
parser.add_argument("--save_num", help="per n passwords generated save once", default=20000000, type=int) # số lượng mật khẩu được sinh ra mỗi lần lưu
parser.add_argument("--batch_size", help="generate batch size", default=5000, type=int) # kích thước lô sinh mật khẩu
parser.add_argument("--gpu_num", help="gpu num", default=1, type=int) # số lượng GPU sử dụng
parser.add_argument("--gpu_index", help="Starting GPU index", default=0, type=int) # chỉ số GPU bắt đầu từ đâu (thường là 0)
args = parser.parse_args()

# Đây là từ điển chứa số lượng ký tự khác nhau cho mỗi loại ký tự trong mật khẩu.
BRUTE_DICT = {'L':52, 'N':10, 'S':32}   # L has 52 different letters, N has 10 different numbers and S has 32.


# the span of three types adhere to vocab.json
# Đây là từ điển chứa các khoảng của ba loại ký tự trong vocab.json.
TYPE_ID_DICT = {'L':(51, 103),
                'N':(41, 51),
                'S':(103, 135),
                }

model_path = args.model_path # đường dẫn đến mô hình GPT-2 đã được huấn luyện
vocab_file = args.vocabfile_path # đường dẫn đến tệp vocab.json
pattern_file = args.pattern_path # đường dẫn đến tệp chứa các mẫu mật khẩu và tỷ lệ của chúng
output_path = args.output_path # đường dẫn đến thư mục đầu ra để lưu trữ mật khẩu đã sinh

n = args.generate_num # số lượng mật khẩu cần sinh
save_num = args.save_num # số lượng mật khẩu được sinh ra mỗi lần lưu
batch_size = args.batch_size # kích thước lô sinh mật khẩu
gpu_num = args.gpu_num # số lượng GPU sử dụng
gpu_index = args.gpu_index # chỉ số GPU bắt đầu từ đâu (thường là 0)

# create new folder to store generation passwords
output_path = output_path + str(n) + '/' # đường dẫn đến thư mục đầu ra để lưu trữ mật khẩu đã sinh
folder = os.path.exists(output_path) # kiểm tra xem thư mục đã tồn tại chưa
if not folder:
    os.makedirs(output_path)

class KeywordsStoppingCriteria(StoppingCriteria):
    """
    Class này được sử dụng để dừng quá trình sinh mật khẩu khi một từ khóa cụ thể được phát hiện trong đầu ra.
    Nó kế thừa từ lớp StoppingCriteria của thư viện transformers.
    """
    def __init__(self, keywords_ids:list):
        """
        Khởi tạo danh sách các từ khóa cần dừng quá trình sinh mật khẩu.
        
        :param keywords_ids: danh sách các từ khóa cần dừng quá trình sinh mật khẩu
        """
        self.keywords = keywords_ids # danh sách các từ khóa cần dừng quá trình sinh mật khẩu

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Kiểm tra xem đầu ra đã chứa từ khóa nào trong danh sách từ khóa hay chưa.
        Nếu có, trả về True để dừng quá trình sinh mật khẩu.
        """
        if input_ids[0][-1] in self.keywords:
            return True
        return False


class SplitBigTask2SmallTask():
    """
    Class này được sử dụng để chia nhỏ các tác vụ lớn thành các tác vụ nhỏ hơn để sinh mật khẩu.
    Nó sử dụng mô hình GPT-2 để sinh ra mật khẩu dựa trên các mẫu đã cho.
    """
    def __init__(self, pcfg_pattern, gen_num, device, tokenizer) -> None:
        """
        Khởi tạo các biến cần thiết cho việc sinh mật khẩu.
        
        :param pcfg_pattern: mẫu mật khẩu cần sinh
        :param gen_num: số lượng mật khẩu cần sinh
        :param device: thiết bị sử dụng (CPU hoặc GPU)
        :param tokenizer: bộ mã hóa được sử dụng để mã hóa và giải mã mật khẩu
        
        """
        self.tasks_list = [] # danh sách các tác vụ cần thực hiện
        
        self.pcfg_pattern = pcfg_pattern # mẫu mật khẩu cần sinh
        # self.gen_num = gen_num
        self.device = device # thiết bị sử dụng (CPU hoặc GPU)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device) # tải mô hình GPT-2 đã được huấn luyện
        self.tokenizer = tokenizer # bộ mã hóa được sử dụng để mã hóa và giải mã mật khẩu
        init_input_ids = tokenizer.encode_forgen(pcfg_pattern) # mã hóa mẫu mật khẩu
        init_input_ids = torch.concat([init_input_ids, torch.tensor([tokenizer.sep_token_id])]).view(1, -1) # thêm token <SEP> vào đầu vào

        self.patterns_list = pcfg_pattern.split(' ') # tách mẫu mật khẩu thành các phần tử dựa trên khoảng trắng
        self.type_list = [] # danh sách các loại ký tự trong mật khẩu
        for pattern in self.patterns_list: # lặp qua từng phần tử trong mẫu mật khẩu
                char_type = pattern[:1] # lấy loại ký tự (L, N hoặc S) từ phần tử
                length = pattern[1:] # lấy độ dài của phần tử
                for i in range(int(length)): # lặp qua độ dài của phần tử
                    self.type_list.append(char_type) # thêm loại ký tự vào danh sách loại ký tự
        self.prefix_length = len(self.patterns_list) + 2 # 2: bos + sep 
        
        max_gen_num = self.judge_gen_num_overflow() # kiểm tra xem số lượng mật khẩu cần sinh có vượt quá giới hạn hay không
        if max_gen_num < gen_num: # nếu số lượng mật khẩu cần sinh vượt quá giới hạn
            gen_num = max_gen_num # đặt lại số lượng mật khẩu cần sinh về giới hạn tối đa
        self.tasks_list.append((init_input_ids, gen_num)) # thêm tác vụ đầu tiên vào danh sách tác vụ cần thực hiện
        self.gen_passwords = [] # danh sách các mật khẩu đã sinh

        
    def __call__(self):
        """
        Hàm này được gọi để thực hiện việc sinh mật khẩu.
        Nó sẽ lặp qua danh sách các tác vụ và thực hiện việc sinh mật khẩu dựa trên các mẫu đã cho.
        """
        more_gen_num = 0 # biến này được sử dụng để theo dõi số lượng mật khẩu cần sinh thêm
        while(len(self.tasks_list) != 0): # lặp qua danh sách các tác vụ
            (input_ids, gen_num) = self.tasks_list.pop() # lấy tác vụ đầu tiên trong danh sách
            if len(input_ids[0]) == self.prefix_length + len(self.type_list): # nếu độ dài của đầu vào bằng độ dài của mẫu mật khẩu cộng với độ dài của danh sách loại ký tự
                self.gen_passwords.append(self.tokenizer.decode(input_ids[0]).split(' ')[1]) # giải mã đầu vào và thêm mật khẩu vào danh sách mật khẩu đã sinh
                more_gen_num = gen_num - 1  # giảm số lượng mật khẩu cần sinh thêm đi 1 vì đã sinh được 1 mật khẩu
                continue
            gen_num = gen_num + more_gen_num # cập nhật số lượng mật khẩu cần sinh thêm
            if gen_num <= batch_size: # nếu số lượng mật khẩu cần sinh nhỏ hơn hoặc bằng kích thước lô
                new_passwords = directly_gen(self.tokenizer, self.device, input_ids, gen_num) # sinh mật khẩu trực tiếp bằng cách sử dụng mô hình GPT-2
                new_passwords_num = len(new_passwords) # số lượng mật khẩu đã sinh được
                self.gen_passwords.extend(new_passwords) # thêm mật khẩu đã sinh vào danh sách mật khẩu đã sinh
                more_gen_num = gen_num - new_passwords_num # cập nhật số lượng mật khẩu cần sinh thêm
            else: # nếu số lượng mật khẩu cần sinh lớn hơn kích thước lô
                next_ids, next_probs = self.get_predict_probability_from_model(input_ids.to(self.device)) # lấy xác suất dự đoán từ mô hình GPT-2
                next_gen_num = next_probs * gen_num # tính toán số lượng mật khẩu cần sinh cho từng phần tử trong đầu vào
                filtered_gen_num = next_gen_num[next_gen_num>=1].view(-1,1) # lọc các phần tử có số lượng mật khẩu cần sinh lớn hơn hoặc bằng 1
                remain_id_num = len(filtered_gen_num) # số lượng phần tử còn lại trong đầu vào
                next_ids = next_ids[:,:remain_id_num] # lọc các phần tử trong đầu vào
                next_probs = next_probs[:,:remain_id_num] # lọc xác suất dự đoán cho các phần tử còn lại
                sum_prob = next_probs.sum() # tính tổng xác suất dự đoán của các phần tử còn lại
                next_probs = next_probs/sum_prob # chuẩn hóa xác suất dự đoán
                next_gen_num = next_probs * gen_num # tính toán số lượng mật khẩu cần sinh cho từng phần tử trong đầu vào
                
                for i in range(remain_id_num): # lặp qua từng phần tử trong đầu vào
                    new_input_ids = torch.cat([input_ids, next_ids[:,i:i+1]], dim=1) # thêm phần tử vào đầu vào
                    new_gen_num = int(next_gen_num[0][i]) # lấy số lượng mật khẩu cần sinh cho phần tử đó
                    self.tasks_list.append((new_input_ids, new_gen_num)) # thêm tác vụ mới vào danh sách tác vụ cần thực hiện
                more_gen_num = 0 # đặt lại số lượng mật khẩu cần sinh thêm về 0
        
        return self.gen_passwords


    def get_predict_probability_from_model(self, input_ids):
        cur_type = self.type_list[len(input_ids[0])-self.prefix_length]
        with torch.no_grad():
            output = self.model(input_ids=input_ids)
            next_token_logits = output.logits[:, -1, :]
            
            type_id_pair = TYPE_ID_DICT[cur_type]

            selected_logits = next_token_logits[:, type_id_pair[0]:type_id_pair[1]]
            selected_softmax = torch.softmax(selected_logits, dim=-1)
            sorted_indices = torch.argsort(selected_softmax, descending=True, dim=-1)
            
            sorted_indexes = sorted_indices + type_id_pair[0]
            sorted_softmax = selected_softmax[:, sorted_indices[0]]
            return sorted_indexes.cpu(), sorted_softmax.cpu()
    

    def judge_gen_num_overflow(self) -> int:
        total = 1
        for _ in self.type_list:
            total = total * BRUTE_DICT[_]
        return total
    
 
def directly_gen(tokenizer, device, input_ids, gen_num):
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    passwords = []

    stop_ids = [tokenizer.pad_token_id]
    stop_criteria = KeywordsStoppingCriteria(stop_ids)
    
    outputs = model.generate(
        input_ids= input_ids.view([1,-1]).to(device),
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
        max_new_tokens=13, 
        do_sample=True, 
        num_return_sequences=gen_num,
        )
    
    outputs = tokenizer.batch_decode(outputs)
    for output in outputs:
        passwords.append(output.split(' ')[1])
    passwords = set(passwords)
    return [*passwords,]
        

def single_gpu_task(task_list, gpu_id, tokenizer):
    gened_passwords = []
    output_count = 1
    finished_task_count = 0
    total_task_num = len(task_list)
    more_gen_num = 0
    while(len(task_list) != 0):
        (pcfg_pattern, num) = task_list.pop()
        num = num + more_gen_num
        print(f'[{finished_task_count}/{total_task_num}] cuda:{gpu_id}\tGenerating {pcfg_pattern}: {num}')
        if num <= batch_size:
            input_ids = tokenizer.encode_forgen(pcfg_pattern)
            input_ids = torch.concat([input_ids, torch.tensor([tokenizer.sep_token_id])])
            new_passwords = directly_gen(tokenizer, 'cuda:'+str(gpu_id), input_ids, num)
        else:
            split2small = SplitBigTask2SmallTask(pcfg_pattern=pcfg_pattern,
                                                 gen_num=num,
                                                 device='cuda:'+str(gpu_id),
                                                 tokenizer=tokenizer)
            new_passwords = split2small()
        
        gened_num = len(new_passwords)
        more_gen_num = num - gened_num
        gened_passwords.extend(new_passwords)
        finished_task_count += 1
        print(f'[{finished_task_count}/{total_task_num}] cuda:{gpu_id}\tActually generated {pcfg_pattern}: {gened_num}\t(diff {num-gened_num})')
        
        while len(gened_passwords) > save_num:
            output_passwords = gened_passwords[:save_num]
            file_path = output_path +'DC-GEN-[cuda:'+ str(gpu_id) + ']-'+str(output_count)+'.txt'
            f = open(file_path, 'w', encoding='utf-8', errors='ignore')
            for password in output_passwords:
                f.write(password+'\n')
            f.close()
            output_count += 1
            gened_passwords = gened_passwords[save_num:]
            print(f'===> File saved in {file_path}.')

    if len(gened_passwords) != 0:
        file_path = output_path + 'DC-GEN-[cuda:'+ str(gpu_id) + ']-last.txt'
        f = open(file_path, 'w', encoding='utf-8', errors='ignore')
        for password in gened_passwords:
            f.write(password+'\n')
        f.close()
        print(f'===> File saved in {file_path}.')


def prepare_task_list(df, gpu_num):
    threshold = 100
    threshold_rate = threshold/n
    filtered_df = df[df['rate'] >= threshold_rate]
    sum_rate = filtered_df['rate'].sum()
    filtered_df['softmax_rate'] = filtered_df['rate']/sum_rate
    
    total_gpu_tasks = []
    for i in range(gpu_num):
        total_gpu_tasks.append([])

    turn = 0
    for row in filtered_df.itertuples():
        pcfg_pattern = row[1]
        num = int(row[3]*n)
        total_gpu_tasks[turn].append((pcfg_pattern, num))
        turn = (turn + 1) % gpu_num

    return total_gpu_tasks


if __name__ == "__main__":
    begin_time = time.time()

    print(f'Load tokenizer.')
    tokenizer = CharTokenizer(vocab_file=vocab_file, 
                                    bos_token="<BOS>",
                                    eos_token="<EOS>",
                                    pad_token="<PAD>",
                                    sep_token="<SEP>",
                                    unk_token="<UNK>"
                                    )
    tokenizer.padding_side = "left"

    print(f'Load patterns.')
    df = pd.read_csv(pattern_file, sep='\t', header=None, names=['pattern', 'rate'])
    total_task_list = prepare_task_list(df, gpu_num)

    # multi threading
    threads = []
    print('*'*30)
    print(f'Generation begin.')
    for i in range(gpu_num):
        thread = threading.Thread(target=single_gpu_task, args=[total_task_list[i], i+gpu_index, tokenizer])
        thread.start()
        threads.append(thread)
    
    for t in threads:
        t.join()
    
    end_time = time.time()
    print('Generation done.')
    print('*'*30)
    print(f'Use time: {end_time-begin_time}')
