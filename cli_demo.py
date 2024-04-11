import os
import torch
import platform
import subprocess
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao-7B-Chat-4K"

def load_model_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        use_fast=False,
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME_OR_PATH)
    return model, tokenizer, generation_config


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.RED + "欢迎使用360智脑大模型，输入进行对话，vim 多行输入，clear 清空历史，stream 开关流式生成，exit 结束。")
    return []


def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name).read()
    return text


def main(stream=True):
    model, tokenizer, generation_config = load_model_tokenizer()

    messages = clear_screen()
    while True:
        try:
            prompt = input(Fore.GREEN + Style.BRIGHT + "\n>用户：" + Style.NORMAL)
            if prompt.strip() == "exit":
                break
            if prompt.strip() == "clear":
                messages = clear_screen()
                continue
            if prompt.strip() == 'vim':
                prompt = vim_input()
                print(prompt)
        except:
            continue

        print(Fore.BLUE + Style.BRIGHT + "\n>助手：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        if stream:
            try:
                messages.append({"role": "user", "content": prompt})
                for response in model.chat(tokenizer=tokenizer, messages=messages, stream=stream, generation_config=generation_config):
                    clear_screen()
                    print(Fore.GREEN + Style.BRIGHT + "\n>用户：" + Style.NORMAL + prompt, flush=True)
                    print(Fore.BLUE + Style.BRIGHT + "\n>助手：" + Style.NORMAL + response, end='', flush=True)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                messages.append({"role": "assistant", "content": response})
            except KeyboardInterrupt:
                pass
            print()
        else:
            messages.append({"role": "user", "content": prompt})
            response = model.chat(tokenizer=tokenizer, messages=messages, stream=stream, generation_config=generation_config)
            messages.append({"role": "assistant", "content": response})
            print(Style.NORMAL + response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
        
    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()