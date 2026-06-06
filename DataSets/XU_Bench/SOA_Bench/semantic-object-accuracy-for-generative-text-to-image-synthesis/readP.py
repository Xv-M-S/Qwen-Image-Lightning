import pickle 

data_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/ground_truth_results/label_00_person_ground_truth.pkl"
data_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/sampled_captions_512res/label_00_person.pkl"

# data_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res_ground_truth/label_00_person_ground_truth.pkl"

with open(data_path, "rb") as f:
    captions = pickle.load(f)
    for i in range(len(captions)):
        print(captions[i])
    
    print(len(captions))