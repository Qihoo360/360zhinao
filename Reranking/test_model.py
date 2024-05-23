
import argparse
from C_MTEB.tasks import *
from flag_dres_model import FlagDRESModel
from flag_models import FlagReranker, FlagRerankerCustom
import torch

if __name__ == "__main__":
    model_name_or_path = "zhinao_1-8b_reranking"
    model = FlagRerankerCustom(model_name_or_path, use_fp16=False)
    inputs=[["天空是什么颜色的","蓝色的"], ["空是什么颜色的","紫色的"],]
    ## 目前返回的是全值域 没有经过sigmoid的分数
    ret = model.compute_score(inputs)
    print(ret)
    ## 如果需要映射到固定值域，可以经过sigmoid 执行下面的代码
    ret = torch.sigmoid(torch.tensor(ret))
    print(ret)



