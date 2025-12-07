# ======================== 环境与依赖配置 ========================
import os
import json
import random
import numpy as np
import mindspore as ms
from tqdm import tqdm
import re
import shutil
from mindspore.dataset import GeneratorDataset
from mindspore.nn import AdamWeightDecay, WarmUpLR
from mindspore.train import Callback
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from mindspore.nn import CrossEntropyLoss
from mindspore.ops import operations as P
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

# 设置MindSpore 2.7.1环境
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device("CPU")  # 使用set_device替代device_target（适配2.7.1警告）
ms.set_seed(42)
np.random.seed(42)
random.seed(42)

# 适配MindSpore 2.7.1的全局设置
ms.set_context(jit_syntax_level=ms.STRICT)
ms.set_recursion_limit(2000)  # 替代max_call_depth

# 配置本地Qwen2.5-0.5B模型路径（请替换为你的实际路径）
LOCAL_MODEL_PATH = os.path.abspath("Qwen2.5-0.5B-Instruct")
print(f"📌 本地模型路径：{LOCAL_MODEL_PATH}")

# 检查路径是否存在
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(
        f"本地模型路径不存在：{LOCAL_MODEL_PATH}\n"
        "请先下载Qwen2.5-0.5B-Instruct模型：\n"
        "地址：https://www.modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct/files"
    )

# ======================== 全局配置（0.5B模型专用） ========================
CONFIG = {
    # Qwen模型配置（0.5B专用）
    "lora_rank": 2,          # 极小秩，最低计算量
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    # 训练配置（CPU流畅运行）
    "max_seq_len": 256,      # 增加序列长度，预留更多生成空间
    "batch_size": 2,         # 0.5B可开2个批量
    "lr": 3e-4,              # 适配小模型的学习率
    "epochs": 10,             # 少量训练轮数
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    # 早停配置
    "early_stop_patience": 2,
    "early_stop_min_delta": 1e-5,
    # 数据配置（平衡效果与速度）
    "data_path": "./health_data.txt",
    "test_size": 20,         # 测试集20条
    "val_size": 20,          # 验证集20条
    "train_sample": 200,     # 训练集仅取200条（CPU快速验证）
    # 输出配置
    "output_dir": "./qwen0.5b_lora_finetune_mindspore271",
    "ckpt_name": "health_advice_qwen0.5b_mindspore271",
}

# ======================== 早停机制（适配MindSpore 2.7.1） ========================
class EarlyStopping(Callback):
    def __init__(self, patience=3, min_delta=1e-5, restore_best_weights=True, verbose=True):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_path = os.path.join(CONFIG["output_dir"], "tmp_best_lora_model.pth")
        self.current_epoch = 0

    def evaluate_loss(self, model, val_dataset, loss_fn):
        """独立的验证损失计算（使用PyTorch损失计算避免类型冲突）"""
        # 切换到PyTorch模型评估
        model.pt_model.eval()
        val_loss = []
        
        with torch.no_grad():
            for batch in val_dataset.create_dict_iterator():
                # MindSpore用int32，转换为PyTorch时转long
                input_ids_np = batch["input_ids"].asnumpy()
                attention_mask_np = batch["attention_mask"].asnumpy()
                labels_np = batch["labels"].asnumpy()
                
                # 先转numpy再转PyTorch long
                input_ids = torch.from_numpy(input_ids_np).long()
                attention_mask = torch.from_numpy(attention_mask_np).long()
                labels = torch.from_numpy(labels_np).long()
                
                # PyTorch前向传播
                outputs = model.pt_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 直接使用PyTorch计算的loss
                loss_np = outputs.loss.cpu().numpy()
                val_loss.append(loss_np)
        
        avg_val_loss = np.mean(val_loss)
        model.pt_model.train()
        return avg_val_loss

    def check_early_stop(self, current_loss, model, epoch):
        """检查早停条件"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                # 保存PyTorch模型权重（更稳定）
                torch.save(model.pt_model.state_dict(), self.best_model_path)
            if self.verbose:
                print(f"✅ 验证损失改进 ({self.best_loss:.6f})，保存最佳LoRA权重")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  验证损失无改进，计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n🛑 触发早停！最佳轮数：{self.best_epoch+1}，最佳损失：{self.best_loss:.6f}")
        return self.early_stop

    def restore_best_model(self, model):
        if self.restore_best_weights and os.path.exists(self.best_model_path):
            try:
                # 恢复PyTorch模型权重
                state_dict = torch.load(self.best_model_path, map_location='cpu')
                model.pt_model.load_state_dict(state_dict)
                
                if self.verbose:
                    print(f"\n✅ 恢复最佳LoRA模型（第{self.best_epoch+1}轮）")
                # 清理临时文件
                os.remove(self.best_model_path)
            except Exception as e:
                print(f"\n❌ 恢复最佳模型失败：{e}")
                pass
        return model

# ======================== 数据处理（适配MindSpore 2.7.1） ========================
def clean_chinese_text(text):
    """轻量化文本清洗，减少CPU计算"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\u4e00-\u9fff0-9，。！？；：""''（）【】《》、·…—]', '', text)
    return text[:100]  # 进一步缩短文本

def load_data_from_txt(file_path):
    """优化数据加载，过滤分类标题行"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # 过滤分类标题行（匹配“X、XXX类（X条）”格式）
            if re.match(r"^[一二三四五六七八九十]+、.*类（\d+ 条）$", line):
                print(f"跳过分类标题行：第{line_num}行 -> {line}")
                continue
            
            parts = line.split("|||")
            if len(parts) != 2:
                print(f"警告：第{line_num}行格式错误 -> {line}")
                continue
            bad_habit = clean_chinese_text(parts[0].strip())
            advice = clean_chinese_text(parts[1].strip())
            if bad_habit and advice:
                # Qwen2.5极简对话模板
                prompt = [
                    {"role": "user", "content": f"用户有不良生活习惯：{bad_habit}，请给出养生建议。"},
                    {"role": "assistant", "content": advice}
                ]
                data.append(prompt)
    print(f"✅ 加载{len(data)}条养生数据")
    return data

def build_dataset():
    """适配0.5B模型的数据集构建，控制数据量"""
    raw_data = load_data_from_txt(CONFIG["data_path"])
    random.shuffle(raw_data)
    
    # 控制训练数据量（CPU快速运行）
    raw_data = raw_data[:CONFIG["train_sample"] + CONFIG["test_size"] + CONFIG["val_size"]]
    
    # 拆分数据集
    test_data = raw_data[:CONFIG["test_size"]]
    val_data = raw_data[CONFIG["test_size"]:CONFIG["test_size"]+CONFIG["val_size"]]
    train_data = raw_data[CONFIG["test_size"]+CONFIG["val_size"]:]
    
    print(f"\n数据拆分（0.5B模型专用）：")
    print(f"  训练集：{len(train_data)}条 | 验证集：{len(val_data)}条 | 测试集：{len(test_data)}条")
    return train_data, val_data, test_data

def format_prompt(data, tokenizer):
    """轻量化Prompt格式化（MindSpore用int32，避免类型警告）"""
    formatted_texts = []
    for item in data:
        # Qwen2.5官方极简模板
        formatted = tokenizer.apply_chat_template(
            item,
            tokenize=False,
            add_generation_prompt=False
        )
        # 快速编码
        encoding = tokenizer(
            formatted,
            truncation=True,
            max_length=CONFIG["max_seq_len"],
            padding="max_length",
            return_tensors="np"  # 使用numpy格式适配MindSpore
        )
        # MindSpore CrossEntropyLoss要求int32
        input_ids_np = encoding["input_ids"].squeeze().astype(np.int32)
        attention_mask_np = encoding["attention_mask"].squeeze().astype(np.int32)
        labels_np = encoding["input_ids"].squeeze().astype(np.int32)
        
        formatted_texts.append({
            "input_ids": ms.Tensor(input_ids_np, dtype=mstype.int32),  # MindSpore要求int32
            "attention_mask": ms.Tensor(attention_mask_np, dtype=mstype.int32),
            "labels": ms.Tensor(labels_np, dtype=mstype.int32),
        })
    return formatted_texts

# ======================== 数据集生成器（MindSpore 2.7.1专用） ========================
class HealthDatasetGenerator:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_ids"], item["attention_mask"], item["labels"]

# ======================== Qwen2.5-0.5B模型加载（直接使用PyTorch+PEFT） ========================
def load_qwen_model_with_lora():
    """加载Qwen2.5-0.5B + LoRA（使用PyTorch核心避免MindSpore兼容问题）"""
    # 加载Tokenizer（强制本地文件）
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True,
        cache_dir=None
    )
    # 兜底设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
    })
    
    # 加载PyTorch版本Qwen模型（避免MindSpore兼容问题）
    print("🔄 加载Qwen2.5-0.5B PyTorch模型...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.float32,  # 替换torch_dtype为dtype（解决警告）
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map="cpu"
    )
    
    # 冻结基座模型
    for param in model.parameters():
        param.requires_grad = False
    
    # 配置LoRA（0.5B专用）
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
        inference_mode=False,
    )

    # 应用LoRA
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    
    # 创建兼容MindSpore的包装器（修复modules属性和类型转换问题）
    class QwenCompatibleWrapper:
        def __init__(self, pt_model, tokenizer):
            self.pt_model = pt_model
            self.tokenizer = tokenizer
            # 添加modules属性支持PEFT
            self.modules = pt_model.modules
        
        def set_train(self, mode):
            if mode:
                self.pt_model.train()
            else:
                self.pt_model.eval()
        
        def get_parameters(self):
            """适配MindSpore的参数获取接口"""
            return [p for p in self.pt_model.parameters() if p.requires_grad]
        
        def parameters_dict(self):
            """适配参数字典接口"""
            return {name: param for name, param in self.pt_model.named_parameters()}
        
        def __call__(self, input_ids, attention_mask=None, labels=None):
            """前向传播（修复astype错误：先转numpy再转PyTorch）"""
            # 正确的类型转换流程：MindSpore Tensor -> numpy -> PyTorch Tensor
            input_ids_np = input_ids.asnumpy()
            attention_mask_np = attention_mask.asnumpy() if attention_mask is not None else None
            labels_np = labels.asnumpy() if labels is not None else None
            
            # 转换为PyTorch long类型
            pt_input_ids = torch.from_numpy(input_ids_np).long()
            pt_attention_mask = torch.from_numpy(attention_mask_np).long() if attention_mask_np is not None else None
            pt_labels = torch.from_numpy(labels_np).long() if labels_np is not None else None
            
            outputs = self.pt_model(
                input_ids=pt_input_ids,
                attention_mask=pt_attention_mask,
                labels=pt_labels
            )
            
            # 返回兼容格式
            logits_np = outputs.logits.detach().cpu().numpy()
            loss_np = outputs.loss.detach().cpu().numpy() if outputs.loss is not None else np.array(0.0)
            
            # 确保loss不为None
            if loss_np is None:
                loss_np = np.array(0.0)
            
            return ms.Tensor(logits_np, mstype.float32), ms.Tensor(loss_np, mstype.float32)
        
        def generate(self, input_ids, attention_mask=None, **kwargs):
            """生成接口（修复类型转换）"""
            # 正确转换流程
            input_ids_np = input_ids.asnumpy()
            attention_mask_np = attention_mask.asnumpy() if attention_mask is not None else None
            
            # 转换为PyTorch long
            pt_input_ids = torch.from_numpy(input_ids_np).long()
            pt_attention_mask = torch.from_numpy(attention_mask_np).long() if attention_mask_np is not None else None
            
            with torch.no_grad():
                outputs = self.pt_model.generate(
                    input_ids=pt_input_ids,
                    attention_mask=pt_attention_mask,** kwargs
                )
            
            # PyTorch long -> numpy int32 -> MindSpore int32
            outputs_np = outputs.cpu().numpy().astype(np.int32)
            return ms.Tensor(outputs_np, mstype.int32)
    
    return QwenCompatibleWrapper(lora_model, tokenizer), tokenizer

# ======================== 训练逻辑（完全使用PyTorch损失计算避免类型冲突） ========================
def train_lora_model():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    train_data, val_data, test_data = build_dataset()
    model, tokenizer = load_qwen_model_with_lora()
    
    # 格式化数据
    train_formatted = format_prompt(train_data, tokenizer)
    val_formatted = format_prompt(val_data, tokenizer)
    test_formatted = format_prompt(test_data, tokenizer)
    
    # 构建MindSpore 2.7.1数据集
    def create_ms_dataset(formatted_data, shuffle=True):
        generator = HealthDatasetGenerator(formatted_data)
        dataset = GeneratorDataset(
            source=generator,
            column_names=["input_ids", "attention_mask", "labels"],
            shuffle=shuffle
        )
        # MindSpore 2.7.1的BatchDataset没有prefetch方法，移除该调用
        dataset = dataset.batch(CONFIG["batch_size"], drop_remainder=True)
        return dataset
    
    train_dataset = create_ms_dataset(train_formatted, shuffle=True)
    val_dataset = create_ms_dataset(val_formatted, shuffle=False)
    test_dataset = create_ms_dataset(test_formatted, shuffle=False)
    
    # 使用PyTorch损失函数（避免MindSpore类型冲突）
    pt_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # 使用PyTorch优化器（更稳定）
    optimizer = torch.optim.AdamW(
        model.pt_model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        eps=1e-8
    )
    
    # 学习率调度器
    total_steps = train_dataset.get_dataset_size() * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 早停实例
    early_stopper = EarlyStopping(
        patience=CONFIG["early_stop_patience"],
        min_delta=CONFIG["early_stop_min_delta"]
    )
    
    # 训练循环
    best_val_loss = float('inf')
    model.set_train(True)
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}]")
        
        # 训练阶段
        train_loss = []
        pbar = tqdm(train_dataset.create_dict_iterator(), desc=f"Training")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # 前向传播
            logits_ms, loss_ms = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            # 使用PyTorch计算损失（避免类型冲突）
            loss_pt = torch.tensor(loss_ms.asnumpy(), requires_grad=True)
            
            # 反向传播
            loss_pt.backward()
            torch.nn.utils.clip_grad_norm_(model.pt_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            loss_val = loss_pt.item()
            train_loss.append(loss_val)
            pbar.set_postfix({"loss": f"{loss_val:.6f}"})
        
        avg_train_loss = np.mean(train_loss)
        
        # 验证阶段（使用PyTorch计算损失）
        avg_val_loss = early_stopper.evaluate_loss(model, val_dataset, pt_loss_fn)
        
        # 测试阶段（使用PyTorch计算损失避免类型冲突）
        model.set_train(False)
        test_loss = []
        
        with torch.no_grad():
            for batch in test_dataset.create_dict_iterator():
                # 前向传播获取logits
                logits_ms, _ = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                # 正确转换为PyTorch张量计算损失
                logits_np = logits_ms.asnumpy()
                labels_np = batch["labels"].asnumpy()
                
                logits_pt = torch.from_numpy(logits_np).float()
                labels_pt = torch.from_numpy(labels_np).long()
                
                # 计算损失
                loss = pt_loss_fn(
                    logits_pt.reshape(-1, logits_pt.shape[-1]),
                    labels_pt.reshape(-1)
                )
                test_loss.append(loss.cpu().numpy())
        
        avg_test_loss = np.mean(test_loss)
        model.set_train(True)
        
        # 检查早停
        if early_stopper.check_early_stop(avg_val_loss, model, epoch):
            break
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['ckpt_name']}_best.pth")
            torch.save(model.pt_model.state_dict(), save_path)
        
        # 打印日志
        print(f"  训练损失: {avg_train_loss:.6f}")
        print(f"  验证损失: {avg_val_loss:.6f}")
        print(f"  测试损失: {avg_test_loss:.6f}")
    
    # 恢复最佳模型
    model = early_stopper.restore_best_model(model)
    
    # 保存最终模型
    final_model_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['ckpt_name']}_final")
    os.makedirs(final_model_path, exist_ok=True)
    
    # 保存PEFT模型和tokenizer
    model.pt_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n✅ Qwen2.5-0.5B LoRA微调完成！模型保存至：{final_model_path}")
    return model, tokenizer

# ======================== 推理（适配MindSpore 2.7.1） ========================
def generate_health_advice(model, tokenizer, bad_habit, max_new_tokens=80):
    """优化生成参数，保证输出完整"""
    # 构建极简对话
    chat = [{"role": "user", "content": f"用户有不良生活习惯：{bad_habit}，请给出养生建议。"}]
    # 格式化Prompt
    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )
    # 编码（MindSpore用int32）
    inputs = tokenizer(
        prompt,
        return_tensors="np",
        truncation=True,
        max_length=CONFIG["max_seq_len"] - max_new_tokens
    )
    
    # MindSpore要求int32
    input_ids_np = inputs["input_ids"].astype(np.int32)
    attention_mask_np = inputs["attention_mask"].astype(np.int32)
    
    # 转换为MindSpore张量
    input_ids = ms.Tensor(input_ids_np, dtype=mstype.int32)
    attention_mask = ms.Tensor(attention_mask_np, dtype=mstype.int32)
    
    # 生成回复
    model.set_train(False)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=1,
        no_repeat_ngram_size=3
    )
    
    # 解码并提取回复
    response = tokenizer.decode(outputs.asnumpy().squeeze().tolist(), skip_special_tokens=True)
    advice = response.split("assistant\n")[-1].strip()
    
    # 兜底：保证输出完整
    if len(advice) < 10 or not advice.endswith(("，", "：", "。", "！", "？")):
        advice += "\n此外，建议保持规律作息、均衡饮食，适度运动，增强身体抵抗力。"
    
    return advice if advice else "建议规律作息，合理饮食，适度运动，保持良好的生活习惯。"

# ======================== 主函数 ========================
if __name__ == "__main__":
    print("=" * 80)
    print(" Qwen2.5-0.5B-Instruct（MindSpore 2.7.1+CPU）LoRA 养生建议微调")
    print("=" * 80)
    
    # 1. 微调模型
    lora_model, tokenizer = train_lora_model()
    
    # 2. 测试生成
    test_cases = [
        "每天熬夜到凌晨1点，早上7点起床",
        "久坐办公室，几乎不运动",
        "经常吃辛辣食物，容易上火",
        "长期不吃早餐，饮食不规律"
    ]
    
    print("\n===== 养生建议生成测试 =====")
    for case in test_cases:
        advice = generate_health_advice(lora_model, tokenizer, case)
        print(f"\n❌ 不良习惯：{case}")
        print(f"✅ 养生建议：{advice}")
        print("-" * 80)
    
    # 3. 加载微调后模型推理
    print("\n===== 加载微调模型推理 =====")
    # 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.float32,
        local_files_only=True,
        device_map="cpu"
    )
    
    # 加载LoRA权重
    from peft import PeftModel
    finetuned_model_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['ckpt_name']}_final")
    finetuned_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
    
    # 创建兼容包装器（修复类型转换）
    class InferenceWrapper:
        def __init__(self, model, tokenizer):
            self.pt_model = model
            self.tokenizer = tokenizer
        
        def set_train(self, mode):
            if mode:
                self.pt_model.train()
            else:
                self.pt_model.eval()
        
        def generate(self, input_ids, attention_mask=None, **kwargs):
            # 正确的类型转换流程
            input_ids_np = input_ids.asnumpy()
            attention_mask_np = attention_mask.asnumpy() if attention_mask is not None else None
            
            # MindSpore int32 -> PyTorch long
            pt_input_ids = torch.from_numpy(input_ids_np).long()
            pt_attention_mask = torch.from_numpy(attention_mask_np).long() if attention_mask_np is not None else None
            
            with torch.no_grad():
                outputs = self.pt_model.generate(
                    input_ids=pt_input_ids,
                    attention_mask=pt_attention_mask,** kwargs
                )
            
            # PyTorch long -> MindSpore int32
            outputs_np = outputs.cpu().numpy().astype(np.int32)
            return ms.Tensor(outputs_np, mstype.int32)
    
    # 加载Tokenizer
    infer_tokenizer = AutoTokenizer.from_pretrained(
        finetuned_model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 最终测试
    test_advice = generate_health_advice(
        InferenceWrapper(finetuned_model, infer_tokenizer),
        infer_tokenizer,
        "换季容易感冒，免疫力差"
    )
    print(f"\n❌ 不良习惯：换季容易感冒，免疫力差")
    print(f"✅ 养生建议：{test_advice}")
    print("=" * 80)