# This file aims to train a PagPassGPT.
"""
File này nhằm mục đích huấn luyện một PagPassGPT. 
"""
from tokenizer.char_tokenizer import CharTokenizer # Thư viện này dùng để mã hóa văn bản thành các token
from transformers import DataCollatorForLanguageModeling # Thư viện này dùng để tạo các batch dữ liệu cho mô hình
from datasets import load_dataset # Thư viện này dùng để tải và xử lý dữ liệu
from transformers import GPT2Config # Thư viện này dùng để cấu hình mô hình GPT-2
from transformers import GPT2LMHeadModel # Thư viện này dùng để tạo mô hình GPT-2 với đầu ra là một chuỗi văn bản
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback # Thư viện này dùng để huấn luyện mô hình
import time
import argparse # Thư viện này dùng để phân tích các tham số đầu vào từ dòng lệnh

parser = argparse.ArgumentParser() # Tạo một đối tượng ArgumentParser để phân tích các tham số đầu vào từ dòng lệnh
# file path parameter setting
parser.add_argument("--dataset_path", help="path of preprocessed train dataset", type=str, required=True) # Đường dẫn đến tập dữ liệu đã được xử lý trước
parser.add_argument("--vocabfile_path", help="path of vocab file", type=str, default="./tokenizer/vocab.json") # Đường dẫn đến tệp vocab
parser.add_argument("--model_path", help="directory to save model", type=str, default="./model/") # Đường dẫn để lưu mô hình
parser.add_argument("--log_path", help="directory of log", type=str, default="./log/") # Đường dẫn để lưu log
# environment parameter setting
parser.add_argument("--random_seed", help="random seed", type=int, default=42) # Hạt giống ngẫu nhiên để đảm bảo tính tái lập của quá trình huấn luyện
parser.add_argument("--num_processer", help="num of processer (cpu logit cores)", type=int, default=10) # Số lượng bộ xử lý (CPU) để xử lý dữ liệu
# model parameter setting
parser.add_argument("--input_size", help="should be larger than (2*max len of password + 3), default is 32 according to max_len=12", type=int, default=32) # Kích thước đầu vào của mô hình, phải lớn hơn (2*max len của mật khẩu + 3), mặc định là 32 theo max_len=12
parser.add_argument("--embed_size", help="embedding size", type=int, default=384) # Kích thước nhúng của mô hình
parser.add_argument("--layer_num", help="num of layers", type=int, default=12) # Số lượng lớp của mô hình
parser.add_argument("--head_num", help="num of multi head", type=int, default=8) # Số lượng đầu của mô hình multi-head attention
# training parameter setting
parser.add_argument("--epoch_num", help="num of epoch (containing early stop))", type=int, default=30) # Số lượng epoch để huấn luyện mô hình (bao gồm cả dừng sớm)
parser.add_argument("--batch_size", help="batch_size", type=int, default=512) # Kích thước batch để huấn luyện mô hình
parser.add_argument("--eval_step", help="eval model every n steps", type=int, default=2000) # Số bước để đánh giá mô hình, mặc định là 2000
parser.add_argument("--save_step", help="save model every n steps", type=int, default=6000) # Số bước để lưu mô hình, mặc định là 6000
parser.add_argument("--early_stop", help="early stop patience", type=int, default=3) # Số bước để dừng sớm, mặc định là 3

args = parser.parse_args() # Phân tích các tham số đầu vào từ dòng lệnh

train_dataset_path = args.dataset_path # Đường dẫn đến tập dữ liệu đã được xử lý trước
vocab_file = args.vocabfile_path # Đường dẫn đến tệp vocab
model_output_dir = args.model_path # Đường dẫn để lưu mô hình đã huấn luyện
log_dir = args.log_path # Đường dẫn để lưu log

random_seed = args.random_seed # Hạt giống ngẫu nhiên để đảm bảo tính tái lập của quá trình huấn luyện
num_processer = args.num_processer # Số lượng bộ xử lý (CPU) để xử lý dữ liệu

# model params: 14260608
input_size = args.input_size # Kích thước đầu vào của mô hình, phải lớn hơn (2*max len của mật khẩu + 3), mặc định là 32 theo max_len=12
embed_size = args.embed_size # Kích thước nhúng của mô hình
layer_num = args.layer_num # Số lượng lớp của mô hình
head_num = args.head_num # Số lượng đầu của mô hình multi-head attention

epoch_num = args.epoch_num # Số lượng epoch để huấn luyện mô hình (bao gồm cả dừng sớm)
batch_size = args.batch_size # Kích thước batch để huấn luyện mô hình
eval_step = args.eval_step # Số bước để đánh giá mô hình, mặc định là 2000
save_step = args.save_step # Số bước để lưu mô hình, mặc định là 6000
early_stop = args.early_stop # Số bước để dừng sớm, mặc định là 3

print(f'Load tokenizer.')
tokenizer = CharTokenizer(vocab_file=vocab_file, # Tạo một tokenizer từ tệp vocab đã cho
                          bos_token="<BOS>", # Token bắt đầu
                          eos_token="<EOS>", # Token kết thúc
                          sep_token="<SEP>", # Token phân tách
                          unk_token="<UNK>", # Token không xác định
                          pad_token="<PAD>", # Token đệm
                          ) # Tạo một tokenizer từ tệp vocab đã cho


print(f'Load dataset.')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # Tạo một data collator để xử lý dữ liệu đầu vào cho mô hình
train_dataset = load_dataset('text', data_files=train_dataset_path, num_proc=num_processer, split='train') # Tải tập dữ liệu từ tệp văn bản đã cho và chia thành các batch dữ liệu
train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], max_len=input_size, padding=True), batched=True) # Mã hóa các dòng văn bản trong tập dữ liệu thành các token bằng tokenizer đã tạo ở trên

print(f'Split dataset into training set and validation set.') # Chia tập dữ liệu thành tập huấn luyện và tập xác thực
train_dataset= train_dataset.train_test_split(test_size=0.125) # Chia tập dữ liệu thành 2 phần: 80% cho tập huấn luyện và 20% cho tập xác thực
eval_dataset = train_dataset['test'] # Tập xác thực
train_dataset = train_dataset['train'] # Tập huấn luyện

print(f'Load model config.') # Tạo một cấu hình cho mô hình GPT-2
config = GPT2Config(
    vocab_size=tokenizer.vocab_size, # Kích thước từ vựng của mô hình
    n_positions=input_size, # Kích thước đầu vào của mô hình
    n_embd=embed_size, # Kích thước nhúng của mô hình
    n_layer=layer_num, # Số lượng lớp của mô hình
    n_head=head_num, # Số lượng đầu của mô hình multi-head attention
    bos_token_id=tokenizer.bos_token_id, # ID của token bắt đầu
    eos_token_id=tokenizer.eos_token_id, # ID của token kết thúc
    activation_function="gelu_new", # Hàm kích hoạt của mô hình
    resid_pdrop=0.1, # Tỷ lệ dropout cho các lớp residual
    embd_pdrop=0.1, # Tỷ lệ dropout cho các lớp nhúng
    attn_pdrop=0.1, # Tỷ lệ dropout cho các lớp attention
    layer_norm_epsilon=1e-5, # Giá trị epsilon cho lớp chuẩn hóa
    initializer_range=0.02, # Phạm vi khởi tạo cho các trọng số của mô hình
    scale_attn_by_inverse_layer_idx=False, # Có nên điều chỉnh trọng số attention theo chỉ số lớp hay không
    reorder_and_upcast_attn=False, # Có nên sắp xếp lại và nâng cao độ chính xác của trọng số attention hay không
)

model = GPT2LMHeadModel(config=config) # Tạo một mô hình GPT-2 với cấu hình đã tạo ở trên
print(f"Num parameters: {model.num_parameters()}") # In ra số lượng tham số của mô hình
print(model)    


print(f'Load training config.') # Tạo một đối tượng TrainingArguments để cấu hình quá trình huấn luyện
training_args = TrainingArguments(
    output_dir=model_output_dir, # Đường dẫn để lưu mô hình đã huấn luyện
    overwrite_output_dir=True, # Ghi đè lên thư mục đầu ra nếu nó đã tồn tại
    num_train_epochs=epoch_num, # Số lượng epoch để huấn luyện mô hình
    per_device_train_batch_size=batch_size, # Kích thước batch để huấn luyện mô hình
    per_device_eval_batch_size=batch_size, # Kích thước batch để xác thực mô hình
    eval_steps = eval_step, # Số bước để đánh giá mô hình
    save_steps=save_step, # Số bước để lưu mô hình
    save_strategy='steps', # Chiến lược lưu mô hình
    evaluation_strategy='steps', # Chiến lược đánh giá mô hình
    prediction_loss_only=True, # Chỉ dự đoán mất mát trong quá trình đánh giá
    logging_dir=log_dir + time.strftime("%Y%m%d-%H:%M", time.localtime()), # Đường dẫn để lưu log
    seed=random_seed, # Hạt giống ngẫu nhiên để đảm bảo tính tái lập của quá trình huấn luyện
    metric_for_best_model='eval_loss', # Tham số để đánh giá mô hình tốt nhất
    load_best_model_at_end=True, # Tải mô hình tốt nhất ở cuối quá trình huấn luyện
    save_total_limit=1 # Giới hạn số lượng mô hình đã lưu
    )

trainer = Trainer( # Tạo một đối tượng Trainer để huấn luyện mô hình
    model=model, # Mô hình để huấn luyện
    args=training_args, # Các tham số huấn luyện
    data_collator=data_collator, # Data collator để xử lý dữ liệu đầu vào cho mô hình
    train_dataset=train_dataset, # Tập huấn luyện
    eval_dataset=eval_dataset, # Tập xác thực
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop)], # Dừng sớm nếu không có cải thiện trong một số bước nhất định
    
)

print(f'*'*30) # In ra các tham số huấn luyện
print(f'Training begin.') # In ra thông báo bắt đầu huấn luyện
trainer.train() # Bắt đầu quá trình huấn luyện mô hình

trainer.save_model(model_output_dir+"last-step/") # Lưu mô hình đã huấn luyện vào thư mục đã chỉ định
print(f'Model saved in {model_output_dir+"last-step/"}') # In ra thông báo đã lưu mô hình
print(f'*'*30) # In ra các tham số huấn luyện
print(f'Training done.') # In ra thông báo đã hoàn thành quá trình huấn luyện