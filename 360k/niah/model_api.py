import requests
import json


class ModelAPI:
    def __init__(self, url, penalty=0):
        if url[-1] == '/':
            self.url_base = url[:-1]
        else:
            self.url_base = url  # has no suffix
        self.penalty = penalty
        print('repetition penalty:', penalty)

        # determine request type (360 or open-source vllm)
        list_model_response = requests.get(self.url_base + '/v1/models')
        if list_model_response.status_code == 200:  # openai API
            self.is_openai = True
            self.url = self.url_base + '/v1/chat/completions'
            self.model_name = list_model_response.json()['data'][0]['id']
        elif list_model_response.status_code == 404:
            self.is_openai = False
            self.url = self.url_base + '/generate'
        else:
            raise NotImplementedError
        
        print('ModelAPI url:', self.url)
    
    def send_request(self, query):
        if self.is_openai:
            params = {
                        "n":1,
                        "best_of":1,
                        "use_beam_search":False,
                        "temperature":1,
                        "top_p":0.5,
                        "top_k":1,
                        # "do_sample": False,
                        "max_tokens":128,
                        "presence_penalty":self.penalty,
                        "frequency_penalty":self.penalty,
                        # "stream":False,
                        "stop":["<|eot_id|>", "<|endoftext|>", "<|im_end|>"],
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": query
                            }
                        ],
                        "model": self.model_name
                    }
        else:
            params = {
                        "n":1,
                        "best_of":1,
                        "use_beam_search":False,
                        "temperature":1,
                        "top_p":0.5,
                        "top_k":1,
                        "do_sample": False,
                        "max_tokens":128,
                        "presence_penalty":self.penalty,
                        "frequency_penalty":self.penalty,
                        "stream":False,
                        "stop":["<|eot_id|>", "<|endoftext|>", "<|im_end|>"],
                        "message": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": query
                            }
                        ]
                    }
        data = json.dumps(params)
        headers = {'Content-Type': 'application/json'}
        try:
            res = requests.post(self.url, data=data, headers=headers, timeout=1800)  # 1800s = 30min timeout is far more than enough
            if self.is_openai:
                predict = res.json()["choices"][0]['message']['content']
            else:
                predict = json.loads(res.text)["output"][0]
        except Exception as e:
            print(e)
            predict = ''
        return predict
    
    def chat(self, query):
        count = 0
        while count < 1:
            try:
                count += 1
                response = self.send_request(query)
                if response:
                    response = response.replace("<|im_end|>", "").replace("<|im_start|>", "")
                    break
            except Exception as e:
                print('Exception:', e)
        return response  


if __name__ == '__main__':
    api = ModelAPI('http://0.0.0.0:7083')
    print(api.chat('你是谁'))
