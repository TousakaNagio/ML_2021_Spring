!nvidia-smi

from google.colab import drive
drive.mount('/content/gdrive')

! cp -r /content/gdrive/MyDrive/ML_HW8/saved_model/ /content/

"""## Download Dataset"""

# Download link 1
!gdown --id '1znKmX08v9Fygp-dgwo7BKiLIf2qL1FH1' --output hw7_data.zip

# Download Link 2 (if the above link fails) 
# !gdown --id '1pOu3FdPdvzielUZyggeD7KDnVy9iW1uC' --output hw7_data.zip

!unzip -o hw7_data.zip

"""## Install transformers

Documentation for the toolkit:　https://huggingface.co/transformers/
"""

# You are allowed to change version of transformers or use other toolkits
!pip install transformers==4.5.0

"""## Import Packages"""

import json
import numpy as np
import random
import torch
import re
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, AutoTokenizer, TFAutoModel
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup

from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)
		    torch.cuda.manual_seed_all(seed)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True
same_seeds(1126)

# Change "fp16_training" to True to support automatic mixed precision training (fp16)	
fp16_training = True

if fp16_training:
    !pip install accelerate==0.2.0
    from accelerate import Accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device

# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/

"""## Load Model and Tokenizer




 
"""

# model = BertForQuestionAnswering.from_pretrained("hfl/chinese-macbert-base").to(device)
model = BertForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext-large").to(device)
tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

# You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)

"""## Read Data

- Training set: 26935 QA pairs
- Dev set: 3523  QA pairs
- Test set: 3492  QA pairs

- {train/dev/test}_questions:	
  - List of dicts with the following keys:
   - id (int)
   - paragraph_id (int)
   - question_text (string)
   - answer_text (string)
   - answer_start (int)
   - answer_end (int)
- {train/dev/test}_paragraphs: 
  - List of strings
  - paragraph_ids in questions correspond to indexs in paragraphs
  - A paragraph may be used by several questions 
"""

max_len = 14

def trim_data(questions):
  new_questions = []
  for _, q in enumerate(questions):
    if (q['answer_end'] - q['answer_start'] <= max_len):
      new_questions.append(q)
  return new_questions

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

train_raw_questions, train_paragraphs = read_data("hw7_train.json")
train_questions = trim_data(train_raw_questions)
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")

"""## Tokenize Data"""

# Tokenize questions and paragraphs separately
# 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__ 

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

# You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model


get_index = lambda x, l: [i for (y, i) in zip(l, range(len(l))) if y in x]

"""## Dataset and Dataloader"""

class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_paragraph_len = 350
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = 40

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)
    
    def get_ans(self, start_idx, idx, ans_start, ans_end):
      question = self.questions[idx]
      tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]
      answer_start_char = tokenized_paragraph.token_to_word(ans_start)
      answer_end_char = tokenized_paragraph.token_to_word(ans_end)
      return start_idx + answer_start_char, start_idx + answer_end_char

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        spl_index = [0] + get_index([511, 136, 106], tokenized_paragraph.ids)

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token + answer_end_token) // 2
            ans = answer_end_token - answer_start_token
            # pos = round(random.random()*(self.max_paragraph_len - ans))
            start_i ,end_i = 0, -1
            while True:
              if spl_index[end_i] - spl_index[start_i] < self.max_paragraph_len:
                break
              else:
                rand = 2*round(random.random()) - 1
                if spl_index[end_i - 1] < answer_end_token:
                  rand = 1
                if spl_index[start_i + 1] > answer_start_token:
                  rand = -1
                if rand > 0:
                  start_i += rand
                else:
                  end_i += rand
            paragraph_start = spl_index[start_i]
            paragraph_end = spl_index[end_i]

            # paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            # paragraph_end = paragraph_start + self.max_paragraph_len
            
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            # for i in range(0, len(tokenized_paragraph), self.doc_stride):
            j = 0
            for i in range(len(spl_index)):
                while True:
                  if j+1 < len(spl_index):
                    if spl_index[j+1] - spl_index[i] <= self.max_paragraph_len:
                      j += 1
                    else:
                      break
                  else:
                    break
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[spl_index[i] : spl_index[j]] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask

train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = 8

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

"""## Function for Evaluation"""

def evaluate(data, output):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')

"""## Training"""

num_epoch = 1
validation = True
logging_step = 50
train = True
learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

iteration = 8
epoch_steps = len(train_questions) / (train_batch_size * iteration)
max_steps = num_epoch * epoch_steps
scheduler = get_linear_schedule_with_warmup(optimizer, max_steps/15, max_steps)

model.train()

print("Start Training ...")

for epoch in range(num_epoch):
    if train:
      step = 1
      train_loss = train_acc = 0
      for batch_idx, data in tqdm(enumerate(train_loader)):	
          # Load all data into GPU
          data = [i.to(device) for i in data]
          
          # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
          # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
          output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

          # Choose the most probable start position / end position
          start_index = torch.argmax(output.start_logits, dim=1)
          end_index = torch.argmax(output.end_logits, dim=1)
          output.loss = output.loss / iteration
          
          # Prediction is correct only if both start_index and end_index are correct
          train_acc += (((start_index == data[3]) & (end_index == data[4])).float().mean())/iteration
          train_loss += output.loss
          
          if fp16_training:
              accelerator.backward(output.loss)
          else:
              output.loss.backward()
          
          if ((batch_idx + 1) % iteration == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f} \
                | Lr: {optimizer.param_groups[0]['lr']}")
                train_loss = train_acc = 0

          ##### TODO: Apply linear learning rate decay #####
          
          # Print training loss and accuracy over past logging step

    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
                # prediction is correct only if answer text exactly matches
                # word_start, ans_start, ans_end = evaluate(data, output)
                # ans_st_ch, ans_end_ch = dev_set.get_ans(word_start, i, ans_start, ans_end)
                # if (i == 0):
                #   print(data)
                # ans = dev_paragraphs[dev_questions[i]['paragraph_id']][ans_st_ch: ans_end_ch+1]
                # print(ans)
                ans = evaluate(data, output)
                if '[UNK]' in ans:
                  ans = re.search(ans.replace('[UNK]', '.'), dev_paragraphs[dev_questions[i]['paragraph_id']]).group()
                # dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
                if ans[-1] == '》' and ans[0] != '《':
                  ans = '《' + ans
                if ans[-1] == '」' and ans[0] != '「':
                  ans = '「' + ans
                if ans[0] == '「' and ans[-1] != '」':
                  ans = ans + '」'
                if ans[0] == '《' and ans[-1] != '》':
                  ans = ans + '》'
                # print(ans)
                dev_acc += ans == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()

# Save a model and its configuration file to the directory 「saved_model」 
# i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
# Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
# print("Saving Model ...")
# print("step:", step)
# model_save_dir = "saved_model" 
# model.save_pretrained(model_save_dir)

print("Saving Model ...")
model_save_dir = "saved_model" 
model.save_pretrained(model_save_dir)

"""## Testing"""

print("Evaluating Test Set ...")

result = []

model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        result.append(evaluate(data, output))

result_file = "result.csv"
with open(result_file, 'w') as f:	
	  f.write("ID,Answer\n")
	  for i, test_question in enumerate(test_questions):
        # Replace commas in answers with empty strings (since csv is separated by comma)
        # Answers in kaggle are processed in the same way
		    f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")

!cp -r /content/saved_model/ /content/gdrive/MyDrive/ML_HW8/