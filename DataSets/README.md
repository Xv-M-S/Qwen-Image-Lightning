# introduction
the repo is used to generate layout dataset.

# fileDownLoad
HRS_Bench: https://github.com/eslambakr/HRS_benchmark
data_evaluate_LLM: https://github.com/Attention-Refocusing/attention-refocusing/tree/main/data_evaluate_LLM

## data_evaluate_LLM
we borrow the data_evaluate_LLM from https://github.com/Attention-Refocusing/attention-refocusing which has generated the some layout data using gpt-3.5-turbo. List below:

### counting
【data_evaluate_LLM/gpt_generated_box/counting.p】【508条】
【/data_evaluate_LLM/gpt_generated_box/counting_5.p】【496条】
【/data_evaluate_LLM/gpt_generated_box/counting_500_1499.p】【986条】
【/data_evaluate_LLM/gpt_generated_box/counting_1500_2499.p】【982条】
【drawbench】【/data_evaluate_LLM/gpt_generated_box_drawbench/counting.p】【19条】
``` case
two_cups_filled_with_steaming_hot_coffee_sit_side-_5: {'prompt': 'two cups filled with steaming hot coffee sit side-by-side on a wooden table.', '0': {'description': 'cup', 'mask': [128, 200, 242, 288]}, '1': {'description': 'cup', 'mask': [270, 200, 384, 288]}, '2': {'description': 'table', 'mask': [20, 320, 492, 512]}, '3': {'description': 'steam', 'mask': [185, 172, 215, 200]}, '4': {'description': 'steam', 'mask': [327, 172, 357, 200]}}
two_parking_meters_are_side-by-side_both_displayi_2: {'prompt': 'two parking meters are side-by-side, both displaying the same time limit and rate of payment.', '0': {'description': 'parking_meter', 'mask': [100, 220, 200, 420]}, '1': {'description': 'parking_meter', 'mask': [220, 220, 320, 420]}}
two_baseball_bats_are_lying_side-by-side_on_a_gras_2: {'prompt': 'two baseball bats are lying side-by-side on a grassy field, ready to be used for a game.', '0': {'description': 'baseball_bat', 'mask': [60, 230, 160, 290]}, '1': {'description': 'baseball_bat', 'mask': [180, 230, 280, 290]}}

```

### size【bigger\smaller】
【/data_evaluate_LLM/gpt_generated_box/size.p】【374条】

### color 【case:red】
【/data_evaluate_LLM/gpt_generated_box/color.p】【483】

### spatial 【below】【on】【up】
【/data_evaluate_LLM/gpt_generated_box/spatial.p】【401】
【/data_evaluate_LLM/gpt_generated_box/spatial_2.p】【496】
【drawbench】【/data_evaluate_LLM/gpt_generated_box_drawbench/spatial.p】【20条】

## there is some dataset that need we to generate and build, especially for visual text.
we use the prompt provided by HRS_Bench, and use the qwen model to generate the layout.

# usage

## convert the format of data_evaluate_LLM to the format of XU_Bench.

``` python
cd DataSets
python dataConvert.py
```

## generate the layout data of visual text

``` python  
cd DataSets
python main.py
```
