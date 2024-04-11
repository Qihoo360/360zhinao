import sys
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

app = Flask(__name__)

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao-7B-Chat-4K"

class InvalidAPIUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['status_code'] = self.status_code
        rv['msg'] = self.message
        return rv

@app.errorhandler(InvalidAPIUsage)
def invalid_api_usage(e):
    return jsonify(e.to_dict()), e.status_code


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

model, tokenizer, generation_config = load_model_tokenizer()

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completion():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if data.get('max_new_tokens', None) is not None:
            generation_config.max_new_tokens = data.get('max_new_tokens', None)
        if data.get('do_sample', None) is not None:
            generation_config.do_sample = data.get('do_sample', None)
        if data.get('top_k', None) is not None:
            generation_config.top_k = data.get('top_k', None)
        if data.get('top_p', None) is not None:
            generation_config.top_p = data.get('top_p', None)
        if data.get('temperature', None) is not None:
            generation_config.temperature = data.get('temperature', None)
        if data.get('repetition_penalty', None) is not None:
            generation_config.repetition_penalty = data.get('repetition_penalty', None)
        
        print("generation_config: ", generation_config)

        response = model.chat(tokenizer=tokenizer, messages=messages, stream=False, generation_config=generation_config)

        response_data = {
            "model": MODEL_NAME_OR_PATH,
            "choices": [{"message": {"role": "assistant", "content": response}}]
        }

        print(f"response:\n{response_data}")

        return jsonify(response_data)

    except Exception as e:
        raise InvalidAPIUsage(str(e), status_code=500)


if __name__ == '__main__':
    port = 8360
    if len(sys.argv) >= 2:
        port = int(sys.argv[1])

    app.run(host='0.0.0.0', port=port)