export NCCL_DEBUG=DEBUG
export PATH=your_env_path

python eval_C-MTEB.py --model qihoo360/360Zhinao-search --query_instruction_for_retrieval 为这个句子生成表示以用于检索相关文章：
python summarize_results.py
