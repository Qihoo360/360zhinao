import argparse
from flag_dres_model import *
from mteb import MTEB


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="qihoo360/360Zhinao-search", type=str)
    parser.add_argument('--query_instruction_for_retrieval', default="为这个句子生成表示以用于检索相关文章：", type=str)
    parser.add_argument('--pooling_method', default='cls', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = FlagDRESModel(model_name_or_path=args.model,
                          query_instruction_for_retrieval=args.query_instruction_for_retrieval,
                          pooling_method=args.pooling_method,
                          batch_size=256)

    task_names = ['T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval',
                  'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval']

    for task in task_names:
        evaluation = MTEB(tasks=[task], task_langs=['zh'])
        evaluation.run(model, output_folder=f"zh_results/{args.model.split('/')[-1]}")

