from tifascore import tifa_score_benchmark
import json

# We recommend using mplug-large
results = tifa_score_benchmark("mplug-large", "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/tifa_v1.0_question_answers.json", "/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/tifaResGINAME/image_id_mapping.json")

# save the results
with open("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa1_evaluation_result_regentest.json", "w") as f:
    json.dump(results, f, indent=4)