from dotenv import load_dotenv
import os
import sys
import tiktoken
import glob
import json
import random
import requests
# from langchain.evaluation import load_evaluator
# from langchain.chat_models import ChatOpenAI
# from anthropic import AsyncAnthropic, Anthropic
from dotenv import load_dotenv
import numpy as np
# from openai import AsyncOpenAI
import asyncio
from asyncio import Semaphore
from datetime import datetime, timezone
import time
from multiprocessing import Pool
from metrics import qa_f1_score, qa_f1_zh_score
from transformers import AutoTokenizer
from tqdm import tqdm
from model_api import ModelAPI

load_dotenv()


def generate_random_number(num_digits):
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)


class LLMNeedleHaystackTester:
    # OURS_TEMPLATE = "{context} {question} Don't give information outside the document or repeat your findings. Keep your response short and direct."
    RANDOM_NEEDLE_CITIES_EN  = list(set([
        'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
        'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
        'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
        'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
        'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
        'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
        'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
        'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
        'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
        'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
        'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
        'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta'
    ]))
    RANDOM_NEEDLE_CITIES_CN = list(set([
    '芝加哥', '仰光', '安塔那那利佛', '科伦坡', '阿拉木图', '悉尼', '墨西哥城',
    '西雅图', '拉各斯', '阿姆斯特丹', '贝尔格莱德', '开罗', '巴格达', '大马士革', '基加利', '达喀尔',
    '达喀尔', '索非亚', '基加利', '维多利亚', '塔什干', '孟买', '巴塞罗那', '阿拉木图', '安曼',
    '多伦多', '布拉迪斯拉发', '约翰内斯堡', '廷布', '曼谷', '圣地亚哥', '开罗', '旧金山',
    '拉各斯', '阿姆斯特丹', '巴黎', '拉巴特', '圣地亚哥', '哥本哈根', '马德里', '基加利',
    '胡志明市', '萨拉热窝', '德里', '伊斯坦布尔', '胡志明市', '喀土穆', '赫尔辛基',
    '多哈', '伊斯坦布尔', '吉隆坡', '布达佩斯', '上海', '莫斯科', '洛杉矶', '奥斯陆',
    '约翰内斯堡', '柏林', '班加罗尔', '东京', '墨尔本', '巴塞罗那', '芝加哥', '路易港',
    '里斯本', '内罗毕', '坎帕拉', '利马', '马普托', '温哥华', '迪拜', '喀土穆', '雅加达',
    '马德里', '埃里温', '贝鲁特', '雅典', '芝加哥', '巴黎', '布加勒斯特', '哥本哈根', '布鲁塞尔',
    '大马士革', '西雅图', '洛杉矶', '埃里温', '维多利亚', '突尼斯', '阿斯塔纳', '首尔',
    '布宜诺斯艾利斯', '曼谷', '科伦坡', '布鲁塞尔', '喀土穆', '多哈', '旧金山', '维也纳', '雅加达'
    ]))

    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 further_instruct=None,
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 200000,
                 context_lengths_num_intervals = 35,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 35,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "Qwen",
                 Qwen_path = None,
                 Qwen_models = None,
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='gpt-4-1106-preview',
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 evaluation_criterion = 'gpt4',
                 pool_multiplier = 1,
                 question_at_beginning = False,
                 rnd_number_digits = 7,
                 is_cn = False):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.further_instruct = further_instruct
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.evaluation_criterion = evaluation_criterion
        self.pool_multiplier = pool_multiplier
        self.question_at_beginning = question_at_beginning
        self.rnd_number_digits = rnd_number_digits
        self.is_cn = is_cn
        self.RANDOM_NEEDLE_CITIES = self.RANDOM_NEEDLE_CITIES_CN if is_cn else self.RANDOM_NEEDLE_CITIES_EN

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        # if model_provider not in ["OpenAI", "Anthropic"]:
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
        if model_provider == "Anthropic" and "claude" not in model_name:
            raise ValueError("If the model provider is 'Anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")
        
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

        # if not self.openai_api_key and not os.getenv('OPENAI_API_KEY'):
        #     raise ValueError("Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model")
        # else:
        #     self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')

        if self.model_provider == "Anthropic":
            if not self.anthropic_api_key and not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("Either anthropic_api_key must be supplied with init, or ANTHROPIC_API_KEY must be in env.")
            else:
                self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
            
        if not self.model_name:
            raise ValueError("model_name must be provided.")
        
        if model_provider == "OpenAI":
            self.model_to_test = AsyncOpenAI(api_key=self.openai_api_key)
            self.enc = tiktoken.encoding_for_model(self.model_name)
        elif model_provider == "Anthropic":
            self.model_to_test = AsyncAnthropic(api_key=self.anthropic_api_key)
            self.enc = Anthropic().get_tokenizer()
        elif model_provider == "Qwen":
            self.model_to_test = None
            self.model_list = Qwen_models
            self.enc = AutoTokenizer.from_pretrained(Qwen_path, trust_remote_code=True)
        
        self.model_to_test_description = model_name
        # self.evaluation_model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key = self.openai_api_key)
        self.evaluation_model = None

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
    
    def run_test_mp(self, mp=False):
        # Run through each iteration of context_lengths and depths
        p = Pool(len(self.model_list) * self.pool_multiplier)
        tasks = []

        if not mp:
            model = self.model_list[0]
            for context_length in self.context_lengths[::-1]:  # 先看长context结果
            # for context_length in self.context_lengths:
                for depth_percent in self.document_depth_percents:
                    self.evaluate_and_log(context_length, depth_percent, model)
            return

        i = 0
        for context_length in self.context_lengths[::-1]:  # 先看长context结果
        # for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                # task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                model = self.model_list[i % len(self.model_list)]
                task = p.apply_async(self.evaluate_and_log, args=(context_length, depth_percent, model))
                tasks.append(task)
                i += 1

        # Wait for all tasks to complete
        # await asyncio.gather(*tasks)
        for task in tqdm(tasks):
            task.get()

    def generate_prompt(self, context, retrieval_question):
        if self.model_provider == "Anthropic":
            with open('Anthropic_prompt.txt', 'r') as file:
                prompt = file.read()
            return prompt.format(retrieval_question=self.retrieval_question, context=context)
        elif self.model_provider == "OpenAI":
            # Generate the prompt for the Anthropic model
            # Replace the following line with the appropriate prompt structure
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings"
                }
            ]
        elif self.model_provider == "Qwen":
            if self.question_at_beginning:
                raise NotImplementedError
                return self.retrieval_question + '\n\n' + context + '\n\n' + (f"{self.retrieval_question} " + "Don't give information outside the document or repeat your findings" if not self.further_instruct else self.further_instruct)
            else:
                return context + '\n\n' + f"{retrieval_question} " + ("Don't give information outside the document or repeat your findings" if not self.further_instruct else self.further_instruct)

    def evaluate_and_log(self, context_length, depth_percent, model):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        random_city = random.choice(self.RANDOM_NEEDLE_CITIES)
        needle_rnd_number = str(generate_random_number(self.rnd_number_digits))
        needle_this = self.needle.format(city=random_city, rnd_number=needle_rnd_number)
        question_this = self.retrieval_question.format(random_city)

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent, needle_this)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context, question_this)

        test_start_time = time.time()

        # Go see if the model can answer the question to pull out your random fact
        if self.model_provider == "OpenAI":
            response = self.model_to_test.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=300,
                temperature=0
            )
            response = response.choices[0].message.content
        elif self.model_provider == "Anthropic":
            response = self.model_to_test.completions.create(
                model=self.model_name,
                max_tokens_to_sample=300,
                prompt=prompt,
                temperature=0
            )
            response = response.completion
        elif self.model_provider == "Qwen":
            response = model.chat(prompt)

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Compare the reponse to the actual needle you placed
        # score = self.evaluate_response(response)
        score = int(needle_rnd_number in response)
        print()
        print(random_city, needle_rnd_number, needle_this, question_this, response, score)

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            # 'prompt': prompt,  # TODO
            'needle' : needle_this,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts', exist_ok=True)

            with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            lang = 'zh' if self.is_cn else 'en'
            result_path = f'results/{self.model_name.replace(".", "_")}_{lang}'
            if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=True)

            # Save the result to file for retesting
            with open(f'{result_path}/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)

        # if self.seconds_to_sleep_between_completions:
        #     await asyncio.sleep(self.seconds_to_sleep_between_completions)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/' + self.model_name.replace(".", "_")
        if not os.path.exists(results_dir):
            return False
        
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent, needle):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length, needle)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider == "OpenAI":
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        elif self.model_provider == "Qwen":
            return self.enc.encode(text)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length, needle):
        tokens_needle = self.encode_text_to_tokens(needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            signs = '.，。？'
            period_tokens = self.encode_text_to_tokens(signs)

            # Then we iteration backwards until we find the first period
            offset = 0
            while tokens_new_context and tokens_new_context[-1] not in period_tokens and offset < 100:
                insertion_point -= 1
                offset += 1
                tokens_new_context = tokens_context[:insertion_point]
            print('offset:', offset)

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def evaluate_response(self, response):
        accuracy_criteria = {
            "accuracy": """
            Score 1: The answer is completely unrelated to the reference.
            Score 3: The answer has minor relevance but does not align with the reference.
            Score 5: The answer has moderate relevance but contains inaccuracies.
            Score 7: The answer aligns with the reference but has minor omissions.
            Score 10: The answer is completely accurate and aligns perfectly with the reference.
            Only respond with a numberical score
            """
        }

        if self.evaluation_criterion == "gpt4":
            # Using GPT-4 to evaluate
            evaluator = load_evaluator(
                "labeled_score_string",
                criteria=accuracy_criteria,
                llm=self.evaluation_model,
            )

            eval_result = evaluator.evaluate_strings(
                # The models response
                prediction=response,

                # The actual answer
                reference=self.needle,

                # The question asked
                input=self.retrieval_question,
            )
        elif self.evaluation_criterion == "f1":
            score = qa_f1_score(response, self.needle)
            eval_result = {'score': score * 10}
        elif self.evaluation_criterion == "f1_zh":
            score = qa_f1_zh_score(response, self.needle)
            eval_result = {'score': score * 10}

        return int(eval_result['score'])

    def get_context_length_in_tokens(self, context):
        if self.model_provider == "OpenAI" or self.model_provider == "Qwen":
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return len(self.enc.encode(context).ids)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        if isinstance(self.haystack_dir, list):
            for text in self.haystack_dir:
                context += text
                if self.get_context_length_in_tokens(context) > max_context_length:
                    break
            return context

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider == "OpenAI" or self.model_provider == "Qwen":
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider == "OpenAI" or self.model_provider == "Qwen":
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, mp=False):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        # asyncio.run(self.run_test())
        self.run_test_mp(mp)


if __name__ == "__main__":
    model_name = sys.argv[1]
    tokenizer_path = sys.argv[2]
    is_cn = eval(sys.argv[3])
    print(is_cn)

    random.seed(12345)

    url_list = ["http://0.0.0.0:7083",]
                # "http://0.0.0.0:7092/generate",]
                # "http://0.0.0.0:7083/360",
                # "http://0.0.0.0:7084/360",
                # "http://0.0.0.0:7085/360",
                # "http://0.0.0.0:7086/360",
                # "http://0.0.0.0:7087/360",
                # "http://0.0.0.0:7088/360",
                # "http://0.0.0.0:7089/360",
                # "http://0.0.0.0:7090/360"]
    model_list = [ModelAPI(url, penalty=0) for url in url_list]

    # is_cn = True
    if is_cn:
        novels_path = 'novel.json'

        novels = list()
        with open(novels_path, 'r') as f:
            for line in f:
                j = json.loads(line)
                novels.append(j['content'])

        needle = "\n{city}特有的魔法数字是：{rnd_number}。\n"
        retrieval_question = "{}特有的魔法数字是多少？"
        further_instruct = "仅基于上述文档，不要给出上述文档以外的信息。简明扼要地回答。"
        haystack_dir = novels
    else:
        needle = "\nThe special magic {city} number is: {rnd_number}\n"
        retrieval_question = "What is the special magic {} number?"
        further_instruct = "Don't give information outside the document or repeat your findings. Keep your response short and direct."
        haystack_dir = "PaulGrahamEssays"

    ht = LLMNeedleHaystackTester(Qwen_path=tokenizer_path,
                                 #model_name='32k-v21_1_2-e6',
                                 haystack_dir=haystack_dir,
                                 model_name=model_name,
                                 Qwen_models=model_list,
                                 is_cn=is_cn,
                                 needle=needle,
                                 retrieval_question=retrieval_question,
                                 further_instruct=further_instruct,
                                 evaluation_criterion='f1',
                                 context_lengths_min=1024,
                                 context_lengths_max=360000,
                                 context_lengths_num_intervals=10,
                                 document_depth_percent_intervals=21,  # 11对应10等分
                                #  document_depth_percent_min=60,
                                #  document_depth_percent_max=60,
                                 pool_multiplier=1,
                                 save_contexts=False,
                                 question_at_beginning=False,
                                 save_results=True)
                                #  save_results=False)
    ht.start_test(mp=False)
