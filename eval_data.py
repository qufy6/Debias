import os
import pickle
import json
from eval import parse_file


result_path = "./result"

#Alpha  #$2

result_name = '_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis10.pkl'




file_path = os.path.join(result_path, result_name)
with open(file_path, "rb") as file:
        read = pickle.load(file)
print('read data:',len(read))
results = {}
results["intrasentence"] = read

with open(
    f"result/_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis10.json", "w"#2/4#$3
) as f:
    json.dump(results, f, indent=2)
predictions_file = "result/_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis10.json"#3/4#$4
output_file = "Evaluation/_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis10.json"#4/4#$5

gold_file_path = os.path.join("/share/home/wenqingchen/feiyu/RL_debias/data", "test.json")

print("Evaluating StereoSet files:")
print(f" - predictions_file: {predictions_file}")

print()
print(f"Evaluating {predictions_file}...")

parse_file(gold_file_path, predictions_file, output_file)