import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import os
import glob
import re
import sys
from metrics import qa_f1_zh_score


language = sys.argv[1]
result_name = sys.argv[2]
use_score_in_f = eval(sys.argv[3])
folder_path = f'../results/{result_name}_{language}'  # Replace with your folder path


# edit distance
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def sensetime_score(predictions, references):
    if len(predictions) != len(references):
        return {"error": "predictions and references have different lengths"}

    total_score = 0
    details = []
    for prediction, reference in zip(predictions, references):
        prediction = re.sub(r'\s+', '', prediction)
        reference = re.sub(r'\s+', '', reference)
        edit_distance = levenshtein_distance(prediction, reference)
        max_len = max(len(prediction), len(reference))
        score = 100 * (1 - edit_distance / max_len) if max_len != 0 else 100

        detail = {
            "pred": prediction,
            "answer": reference,
            "edit_distance": edit_distance,
            "score": score
        }
        total_score += score
        details.append(detail)

    average_score = total_score / len(predictions) if predictions else 0
    result = {"score": average_score, "details": details}
    return result


# sensetime_score(answers, [correct_answer] * 2)
def internlm2_score(pred, ref):
    res = sensetime_score([pred], [ref])
    return res['score']


def en_score(pred, ref):
    ans_pattern_lower = 'sandwich.+?dolores.+?sunny'
    re_res = re.findall(ans_pattern_lower, pred.lower())
    if re_res:
        return 100
    else:
        return internlm2_score(pred, ref)


def zh_score(pred, ref):
    ans_pattern_lower = '刘秀'
    re_res = re.findall(ans_pattern_lower, pred.lower())
    if re_res:
        return 100
    else:
        ref = "王莽在刘秀的手下工作。"
        return qa_f1_zh_score(pred, ref) * 100


if use_score_in_f:
    score_fn = None
else:
    score_fn = zh_score if language == 'zh' else en_score

# Using glob to find all json files in the directory
print(folder_path)
json_files = glob.glob(f"{folder_path}/*.json")
# print(json_files)

# List to hold the data
data = []

# Iterating through each file and extract the 3 columns we need
for file in json_files:
    with open(file, 'r') as f:
        json_data = json.load(f)
        # Extracting the required fields
        document_depth = json_data.get("depth_percent", None)
        context_length = json_data.get("context_length", None)
        score = json_data.get("score", None)
        score *= 100
        if score_fn is not None:
            if score_fn == qa_f1_zh_score:
                score = score_fn(json_data.get('model_response', ''), "王莽在刘秀的手下工作。") * 100
            elif score_fn:
                score = score_fn(json_data.get('model_response', ''), json_data.get('needle', ''))
        # Appending to the list
        data.append({
            "Document Depth": document_depth,
            "Context Length": context_length,
            "Score": score
        })
        # print(score)

# Creating a DataFrame
df = pd.DataFrame(data)

pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
# pivot_table.iloc[:5, :5]

avg_score = pivot_table.values.mean()

# Create a custom colormap. Go to https://coolors.co/ and pick cool colors
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

# Create the heatmap with better aesthetics
plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
sns.heatmap(
    pivot_table,
    # annot=True,
    fmt="g",
    cmap=cmap,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Score'}
)

# More aesthetics
title_lang = 'Chinese' if language == 'zh' else 'English'
plt.title(f'Pressure Testing {result_name} in {title_lang}. Avg score {avg_score:.2f} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
plt.xlabel('Token Limit')  # X-axis label
plt.ylabel('Depth Percent')  # Y-axis label
plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
plt.tight_layout()  # Fits everything neatly into the figure area

# Show the plot
save_folder = 'fig_zh' if language == 'zh' else 'fig_en'
if score_fn is not None:
    plt.savefig(f'{save_folder}/{result_name}.{score_fn.__name__}.png', bbox_inches='tight')
else:
    plt.savefig(f'{save_folder}/{result_name}.f1.png', bbox_inches='tight')
