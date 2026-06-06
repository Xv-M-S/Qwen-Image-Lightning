import os

# 所有成功生成的目录列表
folders = [
    "Generated_Layouts/001_a pumpkin bread, a cinnamon roll, a cup of hot coc",
    "Generated_Layouts/002_a steaming latte, a snowflake-shaped cookie, a woo",
    "Generated_Layouts/003_a blooming cherry blossom, a wicker basket, a glas",
    "Generated_Layouts/004_a coconut drink, a straw hat, a seashell, with vis",
    "Generated_Layouts/005_a stack of books, a reading lamp, a cup of tea, wi",
    "Generated_Layouts/006_a dumbbell, a yoga mat, a protein shake, with visu",
    "Generated_Layouts/007_a campfire, a tent, a marshmallow skewer, with vis",
    "Generated_Layouts/008_a strawberry cake, a macaron set, a milkshake, wit",
    "Generated_Layouts/009_a Christmas tree, a wrapped gift, a candy cane, wi",
    "Generated_Layouts/010_a champagne glass, a party hat, a confetti popper,",
    "Generated_Layouts/011_a bouquet of roses, a watering can, a vase, with v",
    "Generated_Layouts/012_a croissant, a fried egg, a cup of coffee, with vi",
    "Generated_Layouts/013_a snowboard, a ski goggles, a thermos, with visual",
    "Generated_Layouts/014_a cat, a dog, a pet toy, with visual text _Pet Par",
    "Generated_Layouts/015_a guitar, a microphone, a headphone, with visual t"
]

# 输出汇总文件
output_file = "all_optimized_prompts_poster.txt"

with open(output_file, "w", encoding="utf-8") as out_f:
    for folder in folders:
        prompt_file = os.path.join(folder, "optimized_prompts.txt")
        
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    # 写入汇总文件
                    # out_f.write(f"=== {folder} ===\n")
                    out_f.write(content + "\n")
                    print(f"✅ 已读取：{prompt_file}")
            except Exception as e:
                print(f"❌ 读取失败：{prompt_file}, {e}")
        else:
            print(f"⚠️  文件不存在：{prompt_file}")

print(f"\n🎉 全部汇总完成！文件保存在：{output_file}")