"""
目的：针对特定的配置，自动的执行评测步骤
"""

import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import shutil



os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]


def run_mini_batch_gen_image(save_root: str | Path,
                             script_path: str | Path = "mini_batch_gen_image.py"):
    """
    以脚本方式运行 mini_batch_gen_image.py，并传入 --save_root 参数。
    默认脚本位于当前工作目录，也可以显式指定绝对路径。
    """
    save_root = Path(save_root).expanduser().resolve()
    script_path = Path(script_path).expanduser().resolve()

    cmd = [
        "python",               # 或 python3，视环境而定
        str(script_path),
        "--save_root", str(save_root)
    ]

    print("[Run]", " ".join(cmd))
    subprocess.check_call(cmd)   # 如果失败会抛 CalledProcessError



def run_batch_command(data_dir, save_path):
    """
    运行batchRun.py脚本并传入指定参数
    
    参数:
        data_dir: .p文件的路径
        save_path: 图片保存路径
    """
    # 构建命令列表
    cmd = [
        "python", 
        "batchRun.py",
        "--data_dir", data_dir,
        "--save_path", save_path
    ]
    
    try:
        print(f"开始执行命令: {' '.join(cmd)}")
        if "fid" in save_path:
            print("正在执行FID计算")
            cmd = [
                "python",
                "batchRun.py",
                "--data_dir", data_dir,
                "--save_path", save_path,
                "--global_id_map","/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/global_id_map.json"
            ]
        elif "tifa" in save_path:
            print("正在执行Tifa计算")
            cmd = [
                "python",
                "batchRun.py",
                "--data_dir", data_dir,
                "--save_path", save_path,
                "--global_id_map","/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/caption_to_id.json"
            ]
        
        # 执行命令并实时输出 stdout 和 stderr
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # 打印命令输出
        print("命令执行成功，输出如下:")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，返回码: {e.returncode}")
        print(f"错误输出: {e.stdout}")
        return False
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        return False
    
# 默认常量（按你原命令填写）
DEFAULT_CONFIG = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/configs/Partitioned_COI_RS101_2x.yaml"
DEFAULT_WEIGHTS = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/weight/Partitioned_COI_RS101_2x.pth"
DEMO_PY = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/demo.py"

def run_unidet_demo(input_pattern: str,
                    output_dir: str,
                    pkl_path: str,
                    *,
                    mode: str = "subprocess"):
    """
    mode = 'subprocess' | 'direct'
      subprocess : 启动一条与命令行完全等价的子进程（最稳妥）
      direct     : 直接 import demo 并调 main()（更快，但需保证 demo.py 目录在 sys.path）
    """
    if mode not in {"subprocess", "direct"}:
        raise ValueError("mode 只能是 'subprocess' 或 'direct'")

    # 构造 detectron2 风格参数列表
    argv = [
        "demo.py",
        "--config-file", DEFAULT_CONFIG,
        "--input", input_pattern,
        "--pkl_pth", pkl_path,
        "--output", output_dir,
        "--opts",
        "MODEL.WEIGHTS", DEFAULT_WEIGHTS,
    ]

    cmd = ["python", DEMO_PY, *argv[1:]]   # 去掉 argv[0] 的伪文件名
    
    return cmd

SCRIPT_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/dataGen.py"

def run_dataGen(raw_image_dir: str,
                output_root: str,
                *,
                mode: str = "subprocess"):
    """
    mode = 'subprocess' | 'direct'
      subprocess : 启新进程跑脚本（最稳妥）
      direct     : 把脚本当模块 import 并调 main()（需脚本里写有 main()）
    """
    if mode not in {"subprocess", "direct"}:
        raise ValueError("mode 只能是 'subprocess' 或 'direct'")

    argv = [
        "dataGen.py",
        "--raw-image-dir", str(raw_image_dir),
        "--output-root", str(output_root),
    ]


    cmd = ["python", SCRIPT_PATH, *argv[1:]]
    print("[Run]", " ".join(cmd))
    return cmd

def run_in_conda_env(env_name: str, command: str, save_log: bool = False, log_file: str = None):
    """
    在指定 conda 环境中执行命令
    例：
        run_in_conda_env("xu_bench", "python train.py --lr 1e-4")
    """
    # 用 conda 自带的 bash 接口：conda run
    cmd = f"conda run -n {env_name} {command}"
    print("[Run]", cmd)
    if save_log:
        with open(log_file, 'a+') as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=f)
        print(f"日志已保存到 {log_file}")
    else:
        subprocess.run(cmd, shell=True)

data_dirs = [
    "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/counting.p",
    "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/size.p",
    "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/spatial.p",
    "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/color.p",
    "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/coco_prompt_layout.p",
    "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/qwen_coco_combined.p"
]

save_dirs = [
    "countingResGI",
    "sizeResGI",
    "spatialResGI",
    "colorResGI",
    "fidResGI",
    "tifaResGI"
]
steps = [
    "genImage",
    "getSoaImage",
    "copyImage",
    "detectImage",
    "countingEval",
    "sizeEval",
    "spatialEval",
    "colorEval",
    "clipEval",
    "fidEval",
    "tifaEval",
    "soaEval"
]
if __name__ == '__main__':
    #1. 参数解析
    parser = argparse.ArgumentParser(description="运行batchRun.py命令")
    parser.add_argument("--image_save_path", 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/both_iTt_and_tTi_features",
                      help="图片保存路径")
    image_save_path = parser.parse_args().image_save_path

    if "genImage" in steps:
        # 保障在正确的目录下
        os.chdir('/home/sxm/flux-workspace/Qwen-Image-Lightning')

        # 2.运行Qwen模型生成图片
        for data_dir, save_dir in zip(data_dirs, save_dirs):
            save_path = os.path.join(image_save_path, save_dir)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            run_batch_command(data_dir, save_path)

        # 3.运行mini_batch_gen_image.py
        if "getSoaImage" in steps:
            mini_image_path = os.path.join(image_save_path, "soaResGI")
            run_mini_batch_gen_image(mini_image_path)
 
    if "copyImage" in steps:
    # 4.复制图片到指定位置
        for save_dir in save_dirs:
            src_dir = os.path.join(image_save_path, save_dir)
            print("复制图片从", src_dir)
            dst_dir = os.path.join(image_save_path, save_dir + "NAME")
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for filename in os.listdir(src_dir):
                if "generated" in filename:
                    src_file = os.path.join(src_dir, filename)
                    dst_file = os.path.join(dst_dir, filename)
                    shutil.copy(src_file, dst_file)

    if "detectImage" in steps:
        # 保障在正确的目录下
        os.chdir('/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master')
        # 5. 运行函数获取box
        output_det_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/DetectRes"
        output_pkl_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir"
        for save_dir in save_dirs:
            if save_dir == "colorResGI":
                break  # color部分以及后续的不需要运行unidet
            input_pattern = os.path.join(image_save_path, save_dir + "NAME", "*.png")
            output_dir = os.path.join(output_det_path, save_dir)
            pkl_path = os.path.join(output_pkl_path, save_dir + ".pkl")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cmd = run_unidet_demo(input_pattern, output_dir, pkl_path, mode="subprocess")
            print("运行 UniDet 命令：", " ".join(cmd))
            run_in_conda_env("xu_bench", " ".join(cmd))
    

    # 6. 运行相关的eval函数
    # 6.1 运行counting的eval函数
    output_pkl_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir"
    log_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/autologs/counting.log"
    if "countingEval" in steps:
        os.chdir("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/counting")
        count_pkl_path = os.path.join(output_pkl_path, save_dirs[0] + ".pkl")
        count_cmd = f'python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/counting/calc_counting_acc.py {count_pkl_path} /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/counting_prompts.csv 1'
        run_in_conda_env("xu_bench", count_cmd, True, log_file)

    # 6.2 运行size的eval函数
    log_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/autologs/size.log"
    if "sizeEval" in steps:
        size_pkl_path = os.path.join(output_pkl_path, save_dirs[1] + ".pkl")
        os.chdir("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/compositions")
        size_cmd = f'python calc_size_comp_acc.py {size_pkl_path} /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/size_compositions_prompts.csv 1'
        run_in_conda_env("xu_bench", size_cmd, True, log_file)

    # 6.3 运行spatial的eval函数
    log_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/autologs/spatial.log"
    if "spatialEval" in steps:
        spatial_pkl_path = os.path.join(output_pkl_path, save_dirs[2] + ".pkl")
        os.chdir("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/compositions")
        spatial_cmd = f'python calc_spatial_relation_acc.py {spatial_pkl_path} /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/spatial_compositions_prompts.csv 1'
        run_in_conda_env("xu_bench", spatial_cmd, True, log_file)

    # 6.4 运行color的eval函数
    log_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/autologs/color.log"
    if "colorEval" in steps:
        color_image_path = os.path.join(image_save_path, save_dirs[3] + "NAME")
        os.chdir("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors")
        color_cmd = f'python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/hue_based_color_classifier.py /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/outputData/colorRes /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/colors_composition_prompts.csv {color_image_path}'
        run_in_conda_env("xu_bench", color_cmd, True, log_file)


    # 7 计算clip score
    log_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/autologs/clip.log"
    if "clipEval" in steps:
        os.chdir("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench")
        clip_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/format_data"
        # 7.1 生成image text pairs
        for save_dir in save_dirs:
            if save_dir == "fidResGI":
                break  # color部分以及后续的不需要运行unidet
            input_path = os.path.join(image_save_path, save_dir + "NAME")
            output_path = os.path.join(clip_path, save_dir)
            cmd = run_dataGen(input_path, output_path, mode="subprocess")
            run_in_conda_env("tifaEnv", " ".join(cmd))

            input_txt = os.path.join(output_path, "text")
            input_img = os.path.join(output_path, "image")
            clip_cmd =f'python -m clip_score {input_img} {input_txt}'
            run_in_conda_env("tifaEnv", clip_cmd, True, log_file)

    # 8. 计算fid score
    log_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/autologs/fid.log"
    if "fidEval" in steps:
        os.chdir("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/FID")
        fid_path = os.path.join(image_save_path, "fidResGINAME")
        fid_cmd = f'python fid_score.py --gpu 1 --batch-size 24 --path1 /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/real_images --path2 {fid_path}'
        run_in_conda_env("tifaEnv", fid_cmd, True, log_file)

    # 9. 计算tifa score 
    log_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/autologs/tifa.log"
    if "tifaEval" in steps:
        os.chdir("/home/sxm/flux-workspace/Qwen-Image-Lightning/tifa")
        tifa_data = os.path.join(image_save_path, "tifaResGINAME")
        json_cmd = f'python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/generate_image_json.py --image_folder {tifa_data}'
        run_in_conda_env("tifaEnv", json_cmd)
        json_path = os.path.join(image_save_path, "tifaResGINAME", "image_id_mapping.json")
        tifa_cmd = f'python tifaSampleTest.py --json_path {json_path}'
        run_in_conda_env("tifaEnv", tifa_cmd, True, log_file)


    # 10.计算SOA-I score 和SOA-C score, iou
    if "soaEval" in steps:
        # 10.1 复制出生成的图片
        source_dir = os.path.join(image_save_path, "soaResGI")
        new_folder = os.path.join(image_save_path, "soaResGINAME")
        mv_cmd = f'python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mvFile.py --source_dir {source_dir} --new_folder {new_folder}'
        run_in_conda_env("tifaEnv", mv_cmd)

        # 10.2 复制groundTruth到对应的文件夹
        cp_gt_cmd = f'python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mvBoxFile.py --ground_truth_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res_ground_truth --target_root_dir {new_folder}'
        run_in_conda_env("tifaEnv", cp_gt_cmd)
        # 10.3 运行评估命令
        log_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/autologs/soa.log"
        eval_save_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA/eval_res_temp"
        os.chdir("/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA")
        soa_eval_cmd = f'python calculate_soa.py --images {new_folder} --output {eval_save_path} --iou'
        run_in_conda_env("tifaEnv", soa_eval_cmd, True, log_file)
        shutil.rmtree(eval_save_path) if os.path.exists(eval_save_path) else None


        # 保障在正确的目录下
        os.chdir('/home/sxm/flux-workspace/Qwen-Image-Lightning')






