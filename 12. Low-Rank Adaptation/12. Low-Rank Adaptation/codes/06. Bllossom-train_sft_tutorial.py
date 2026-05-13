import os
import json
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from peft import get_peft_model,LoraConfig,TaskType,PeftModel
from datasets import load_dataset,Dataset,concatenate_datasets
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch

# PROMPT 설정
PROMPT = \
'''당신은 유용한 AI 어시스턴트로, 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다. 

Non-parametric memory는 주어진 외부 Context를 활용하여 답변을 하는 것을 말하며, Parametric memory는 외부 Context 없이 모델 자체의 저장된 지식을 이용하는 것을 의미합니다.

질문에 따라 필요한 Context가 제공될 경우, Non-parametric memory를 기반으로 답변을 제공해야 합니다. 제공된 Context가 없을 경우에는, 당신의 내장된 지식(Parametric memory)을 활용하여 답변을 생성합니다.'''



# 데이터 전처리 함수 
# SFT를 하기 위해서는 2가지의 전처리 함수 또는 클래스를 사용
# 1. formatting_func : 데이터 토큰화 및 IGNORE INDEX(답변을 제외한 System Message 및 유저 Instruction 부분)처리 
def formatting_func(examples):
    input_ids=[]
    labels = []
        
    for ins,inp,ou in zip(examples['instruction'],examples['input'],examples['output']):
        instruction = ins
        response = ou
        context =inp 
        
        messages = [{'role':'system', 'content':f"{PROMPT}\n\ncontext : {context}"},
                    {'role':'user', 'content':instruction}]
        
        # tokenizer.apply_chat_template 사용법 (apply_chat_template을 미리 만들어 놓은 토크나이져만 가능)
        # messages : 리스트안에 {'role':'...' , 'content':'...'}와 같은 딕셔너리 형태
        # tokenize(True or False) : chat template으로 만든 후 토큰화를 진행할것인지 -> 토큰화를 진행할 경우 input_ids 값만 나옴
        # add_generation_prompt (True or False) : chat template 마지막 부분에 assistant 턴을 추가할 것인지 (아래 예시 참조)
        # Example) add_generation_prompt=True
        # messages의 chat template 뒤에 "<|start_header_id|>assistant<|end_header_id|>"가 추가됨 
        instruction_chat= tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        response_chat = tokenizer(response,return_attention_mask=False,add_special_tokens=False)['input_ids']
        
        chat_messages = instruction_chat+response_chat+[tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        
        # -100 = IGNORE INDEX
        # IGNORE INDEX를 하는 이유 : 모델 내에서 어텐션 계산은 진행하지만(프롬프트를 이해하기 위해) IGNORE INDEX 부분에 학습은 진행하지 않게 하기 위해서
        label = ([-100]*len(instruction_chat))+response_chat+[tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        
        input_ids.append(chat_messages)
        labels.append(label)
    
    # 꼭 return은 딕셔너리로 키값은 input_ids 와 labels로 해야 합니다
    # why? -> 허깅페이스 내부 Trainer에서 input_ids, label_id or label ,attention_mask 만 그대로 두고 나머지 컬럼들을 모조리 삭제해버림
    # labels와 같이 레이블 이름을 다르게 사용하고 싶다면? 아래쪽 Coming Soon...
    return {'input_ids':input_ids,'labels':labels}

# 데이터 콜레이터 클래스
# 데이터 콜레이터는 언제 사용될까? -> Trainer에서 학습 들어가기 전 필요없는 컬럼들을 지우고 DataLoader로 로드
# 데이터 콜레이터의 역할은 뭘까? -> 데이터 콜레이터의 역할은 딱 2개로 지정하는 것이 좋아보임 전처리는 formatting_func에서 끝내고 DataCollator는 패딩과 attention_mask 생성이 적절해 보임 (in SFT)
class CustomDataCollator(object):
    def __init__(self,tokenizer,prompt,padding_value,batch_first):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.padding_value=padding_value
        self.batch_first=batch_first

    def __call__(self, examples):
        # [{},{},{}]
        input_ids=[]
        labels = []
        
        for i in range(len(examples)):
            input_ids.append(torch.tensor(examples[i]['input_ids'],dtype=torch.long))
            labels.append(torch.tensor(examples[i]['labels'],dtype=torch.long))
        
        # pad_sequence는 리스트안에 tensor들로 구성되어 있고 제일 긴 텐서를 기준으로 패딩을 진행
        padded_input_ids = pad_sequence(input_ids,padding_value=self.padding_value,batch_first=self.batch_first)
        padded_labels = pad_sequence(labels,padding_value=self.padding_value,batch_first=self.batch_first)
        
        # '.ne' 는 not equal로 padding_value와 동일한 값들만 0으로 처리해버림
        attention_mask = padded_input_ids.ne(self.padding_value)
    
        # 궁금증 : 왜 패딩은 IGNORE INDEX 값으로 하지 않았을까? -> attention_mask 때문에 어텐션 스코어 점수가 0이 되버릴수 있어서 구분하기 위해 다른 값 사용
        return {'input_ids': padded_input_ids, 'labels': padded_labels,'attention_mask':attention_mask}


# model and tokenizer load
model_id = "MLP-KTLim/bllossom3_non_trained"
base_model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,device_map={"":int(os.environ.get('LOCAL_RANK') or 0)})
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='right'

# added lora adapter 
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=2,
#     target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
#     lora_alpha = 128,
#     lora_dropout=0.05,
#     modules_to_save=['embed_tokens','lm_head']
# )
#model = get_peft_model(base_model,lora_config)

model = PeftModel.from_pretrained(base_model,'/home/rag/train/checkpoint/bllossom_expansion_vocab_20/checkpoint-111984')

# dataset load
alpaca_dataset = load_dataset('MLP-KTLim/koalpaca_for_sft',split='train')
lima_en_dataset = load_dataset('MLP-KTLim/cleaned_lima_en',split='train')
lima_ko_dataset = load_dataset('MLP-KTLim/korquad_lima_rv517',split='train').select(range(len(lima_en_dataset)))
lima_dataset = concatenate_datasets([lima_en_dataset,lima_ko_dataset])
gpt_dataset = load_dataset('MLP-KTLim/kosafe_non_sorry',split='train')
wiki_gpt_dataset = load_dataset('MLP-KTLim/wiki_gpt_rag_ko_en',split='train')
merged_dataset = concatenate_datasets([gpt_dataset, alpaca_dataset, lima_dataset, wiki_gpt_dataset])#
train_dataset = merged_dataset.shuffle()

print('데이터셋 길이: ',len(train_dataset))

# Dataset 객체 map 함수가 몇십배는 전처리 속도가 빠름
train_dataset = train_dataset.map(formatting_func,
                                  num_proc=8,
                                  batched=True)

rag_dataset=train_dataset.train_test_split(test_size=int(len(train_dataset)*0.1),seed=42)


# Training Arguments
output_dir='/home/rag/train/checkpoint/we_realize_attn/lima_ek_alpa_eth_wiki-pt20'
training_args = TrainingArguments(
    dataloader_num_workers=12,
    output_dir = output_dir,
    num_train_epochs = 15,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    save_total_limit=5,
    optim='adamw_torch_fused',
    load_best_model_at_end=True,
    logging_strategy='steps',
    logging_steps=30,
    label_names=['labels'], # Coming soon 부분 -> label_id, label 이 아닌 다른 레이블 값을 사용할 경우 여기다가 이름을 넣어주면 됩니다!
    run_name = 'rag_pt20',
    report_to = 'wandb',
    #torch_compile=True # 모델 로드 오래걸리는 반면, vram이 적게 먹음 + DataCollator 쓸때 에러나서 빼니 에러가 사라짐
)

data_collator = CustomDataCollator(tokenizer=tokenizer,
                                   prompt=PROMPT,
                                   padding_value=tokenizer.pad_token_id,
                                   batch_first=True)

trainer = Trainer(
    model=model,
    train_dataset=rag_dataset['train'],
    eval_dataset=rag_dataset['test'],
    args=training_args,
    data_collator=data_collator,
)

trainer.train()