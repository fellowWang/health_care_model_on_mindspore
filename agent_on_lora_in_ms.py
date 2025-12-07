# ======================== 交互式养生建议助手 ========================
import os
import numpy as np
import mindspore as ms
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# ======================== 配置项（需与训练脚本保持一致） ========================
# 1. 基础配置
LOCAL_MODEL_PATH = os.path.abspath("Qwen2.5-0.5B-Instruct")  # 基座模型路径
FINETUNED_MODEL_PATH = os.path.abspath("./qwen0.5b_lora_finetune_mindspore271/health_advice_qwen0.5b_mindspore271_final")  # 训练好的LoRA模型路径
MAX_SEQ_LEN = 256  # 与训练时一致
MAX_NEW_TOKENS = 128  # 生成回复的最大长度
DEVICE = "cpu"

# 2. MindSpore环境配置（保持与训练一致）
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(42)

# ======================== 模型加载工具类 ========================
class HealthAdviceAssistant:
    def __init__(self):
        # 1. 加载Tokenizer
        print("🔄 加载Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL_PATH,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True,
            cache_dir=None
        )
        # 兜底设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "bos_token": "<|endoftext|>",
        })

        # 2. 加载基座模型 + LoRA权重
        print("🔄 加载微调后的LoRA模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map=DEVICE
        )
        # 加载训练好的LoRA权重
        self.finetuned_model = PeftModel.from_pretrained(
            self.base_model,
            FINETUNED_MODEL_PATH,
            local_files_only=True
        )
        self.finetuned_model.eval()  # 推理模式

        # 3. 创建MindSpore-PyTorch兼容包装器
        class InferenceWrapper:
            def __init__(self, pt_model, tokenizer):
                self.pt_model = pt_model
                self.tokenizer = tokenizer

            def set_train(self, mode):
                if mode:
                    self.pt_model.train()
                else:
                    self.pt_model.eval()

            def generate(self, input_ids, attention_mask=None, **kwargs):
                """适配MindSpore张量类型的生成接口"""
                # MindSpore Tensor -> numpy -> PyTorch Tensor
                input_ids_np = input_ids.asnumpy()
                attention_mask_np = attention_mask.asnumpy() if attention_mask is not None else None

                # 转换为PyTorch long类型
                pt_input_ids = torch.from_numpy(input_ids_np).long()
                pt_attention_mask = torch.from_numpy(attention_mask_np).long() if attention_mask_np is not None else None

                # 生成回复（禁用梯度）
                with torch.no_grad():
                    outputs = self.pt_model.generate(
                        input_ids=pt_input_ids,
                        attention_mask=pt_attention_mask,** kwargs
                    )

                # PyTorch -> numpy -> MindSpore Tensor
                outputs_np = outputs.cpu().numpy().astype(np.int32)
                return ms.Tensor(outputs_np, ms.int32)

        self.model_wrapper = InferenceWrapper(self.finetuned_model, self.tokenizer)
        print("✅ 模型加载完成！")

    def generate_advice(self, bad_habit: str) -> str:
        """
        生成养生建议
        :param bad_habit: 用户输入的不良生活习惯
        :return: 格式化的养生建议
        """
        # 1. 构建Qwen2.5对话模板
        chat = [{"role": "user", "content": f"用户有不良生活习惯：{bad_habit}，请给出养生建议。"}]
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. 编码为MindSpore张量（适配训练时的格式）
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            truncation=True,
            max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS
        )
        input_ids_np = inputs["input_ids"].astype(np.int32)
        attention_mask_np = inputs["attention_mask"].astype(np.int32)
        input_ids = ms.Tensor(input_ids_np, ms.int32)
        attention_mask = ms.Tensor(attention_mask_np, ms.int32)

        # 3. 生成回复
        self.model_wrapper.set_train(False)
        outputs = self.model_wrapper.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,        # 控制生成随机性
            top_p=0.9,              # 核采样
            repetition_penalty=1.15, # 避免重复
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            no_repeat_ngram_size=3
        )

        # 4. 解码并提取回复
        response = self.tokenizer.decode(outputs.asnumpy().squeeze().tolist(), skip_special_tokens=True)
        advice = response.split("assistant\n")[-1].strip()

        # 5. 兜底处理（保证回复完整性）
        if len(advice) < 10 or not advice.endswith(("，", "。", "！", "？", "：")):
            advice += "\n💡 补充建议：保持规律作息、均衡饮食，适度运动，增强身体抵抗力。"

        return advice

# ======================== 交互式对话主程序 ========================
def main():
    # 初始化助手
    try:
        assistant = HealthAdviceAssistant()
    except FileNotFoundError as e:
        print(f"❌ 模型文件未找到：{e}")
        print("请确认：")
        print(f"  1. 基座模型路径：{LOCAL_MODEL_PATH} 存在")
        print(f"  2. 微调模型路径：{FINETUNED_MODEL_PATH} 存在")
        return
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 欢迎语
    print("\n" + "="*80)
    print("🎯 养生建议智能助手（基于Qwen2.5-0.5B LoRA微调）")
    print("="*80)
    print("💡 输入不良生活习惯，我会为您提供个性化养生建议")
    print("💡 输入 '退出'/'quit'/'exit' 可结束对话")
    print("="*80 + "\n")

    # 交互式对话循环
    while True:
        # 获取用户输入
        user_input = input("🧑 您的不良生活习惯：").strip()

        # 退出条件
        if user_input.lower() in ["退出", "quit", "exit", "q"]:
            print("👋 感谢使用，祝您身体健康！")
            break

        # 空输入处理
        if not user_input:
            print("⚠️  请输入有效的不良生活习惯描述（例如：每天熬夜到凌晨1点）\n")
            continue

        # 生成并展示建议
        try:
            print("🤖 养生建议：", end="")
            advice = assistant.generate_advice(user_input)
            print(advice + "\n")
        except Exception as e:
            print(f"❌ 生成建议失败：{e}\n")

if __name__ == "__main__":
    main()