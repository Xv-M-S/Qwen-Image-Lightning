import pickle

with open("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/coco_prompt_layout.p", 'rb') as f:
    raw_data = pickle.load(f)

for i in range(len(raw_data)):
    print(raw_data[i])