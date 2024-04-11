import time
import codecs
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Optional, List, Union
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    UsageInfo)
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRA
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor

logger = init_logger(__name__)

from scipy.optimize import minimize
import sympy
import re
import math
def parse_pot_no_stream(inputs):
    try:
        s = re.findall(r'<<(.*?)>>', inputs, re.DOTALL)
        if not s:
            #print("err inputs: ", origin_inputs, flush=True)
            return inputs

        index = 0
        for k in s:
            try:
                if "func" in k:
                    var = k.split("=", 1)
                    try:
                        var[1] = var[1].strip(" ")
                        exec(var[1], globals())
                        ans = func()
                    except:
                        if 'sympy' in var[1]:
                            var[1] = var[1].replace('res[x]', 'res[0][0]').replace('res[y]', 'res[0][1]')
                            exec(var[1], globals())
                            ans = func()
                        pass
                    var_list = [c.strip(" ") for c in var[0].split(",")]
                    if len(var_list) == 1:
                        ans = [ans]

                    for i in range(len(ans)):
                        try:
                            ans[i] = float(ans[i])
                            if abs(ans[i] - int(ans[i])) < 1e-10:
                                ans[i] = str(int(ans[i]))
                        except:
                            pass

                    inputs = inputs.replace("<<"+k+">>", "")
                    for i in range(len(var_list)):
                        inputs = inputs.replace(var_list[i], str(ans[i]))
                    index += 1
                    for c in range(index, len(s)):
                        for i in range(len(var_list)):
                            s[c] = s[c].replace(var_list[i], str(ans[i]))
                else:
                    var = k.replace(" ", "").split("=")
                    var[1] = var[1].replace("eval", "")
                    ans = round(eval(var[1]), 10)
                    ans = float(ans)
                    if abs(ans - int(ans)) < 1e-10:
                        ans = str(int(ans))
                    inputs = inputs.replace("<<"+k+">>", "").replace(var[0], str(ans))
                    index += 1
                    for c in range(index, len(s)):
                        s[c] = s[c].replace(var[0], str(ans))
            except:
                #print("err inputs: ", origin_inputs, flush=True)
                return "抱歉！你的问题无法回答，请重新输入！"
    except Exception as e:
        #print("err inputs: ", origin_inputs, flush=True)
        return "抱歉！你的问题无法回答，请重新输入！"

    return inputs

class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model: str,
                 response_role: str,
                 lora_modules: Optional[List[LoRA]] = None,
                 chat_template=None):
        super().__init__(engine=engine,
                         served_model=served_model,
                         lora_modules=lora_modules)
        self.response_role = response_role
        self._load_chat_template(chat_template)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt)
        except Exception as e:
            logger.error(
                f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))

        request_id = f"cmpl-{random_uuid()}"
        try:
            token_ids = self._validate_prompt_and_tokenize(request,
                                                           prompt=prompt)
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
            guided_decode_logits_processor = (
                await get_guided_decoding_logits_processor(
                    request, self.engine.get_tokenizer()))
            if guided_decode_logits_processor:
                if sampling_params.logits_processors is None:
                    sampling_params.logits_processors = []
                sampling_params.logits_processors.append(
                    guided_decode_logits_processor)
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator = self.engine.generate(prompt, sampling_params,
                                                request_id, token_ids,
                                                lora_request)
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id)
        else:
            return await self.chat_completion_full_generator(
                request, raw_request, result_generator, request_id)

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput], request_id: str
    ) -> Union[ErrorResponse, AsyncGenerator[str, None]]:

        model_name = request.model
        created_time = int(time.monotonic())
        chunk_object_type = "chat.completion.chunk"

        # Send first response for each request.n (index) with the role
        role = self.get_chat_request_role(request)
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role=role),
                logprobs=None,
                finish_reason=None)
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 object=chunk_object_type,
                                                 created=created_time,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]

            if last_msg_content:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=last_msg_content),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        logprobs=None,
                        model=model_name)
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index

                if finish_reason_sent[i]:
                    continue

                delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                top_logprobs = output.logprobs[
                    previous_num_tokens[i]:] if output.logprobs else None

                if request.logprobs:
                    logprobs = self._create_logprobs(
                        token_ids=delta_token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=request.logprobs,
                        initial_text_offset=len(previous_texts[i]),
                    )
                else:
                    logprobs = None

                #delta_text = output.text[len(previous_texts[i]):]
                #previous_texts[i] = output.text
                if not "<<" in output.text:
                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                else :
                    if not ">>" in output.text:
                        delta_text = ""
                        previous_texts[i] = previous_texts[i-1]
                    else :
                        pot_text = parse_pot_no_stream(output.text)
                        if "var" in pot_text[-6:]:
                            delta_text = ""
                            previous_texts[i] = previous_texts[i-1]
                        else :
                            delta_text = pot_text[len(previous_texts[i]):]
                            previous_texts[i] = pot_text

                previous_num_tokens[i] = len(output.token_ids)
                if output.finish_reason is None:
                    # Send token-by-token response for each request.n
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        logprobs=logprobs,
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                else:
                    # Send the finish response for each request.n only once
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=previous_num_tokens[i],
                        total_tokens=prompt_tokens + previous_num_tokens[i],
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        logprobs=logprobs,
                        finish_reason=output.finish_reason)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    if final_usage is not None:
                        chunk.usage = final_usage
                    data = chunk.model_dump_json(exclude_unset=True,
                                                 exclude_none=True)
                    yield f"data: {data}\n\n"
                    finish_reason_sent[i] = True
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
            self, request: ChatCompletionRequest, raw_request: Request,
            result_generator: AsyncIterator[RequestOutput],
            request_id: str) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = request.model
        created_time = int(time.monotonic())
        final_res: RequestOutput = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            top_logprobs = output.logprobs

            if request.logprobs:
                logprobs = self._create_logprobs(
                    token_ids=token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.logprobs,
                )
            else:
                logprobs = None

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=parse_pot_no_stream(output.text)),
                logprobs=logprobs,
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    def _load_chat_template(self, chat_template):
        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    self.tokenizer.chat_template = f.read()
            except OSError:
                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                self.tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info(
                f"Using supplied chat template:\n{self.tokenizer.chat_template}"
            )
        elif self.tokenizer.chat_template is not None:
            logger.info(
                f"Using default chat template:\n{self.tokenizer.chat_template}"
            )
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")
