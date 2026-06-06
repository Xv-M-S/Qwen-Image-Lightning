
data = [
{
    "prompt": "一束粉色玫瑰、一盆白色郁金香、一个木质花架，搭配视觉文本“春日花宴”和“全场8折”，背景为浅粉色。",
    "0": {"description": "粉色玫瑰", "mask": [64, 173, 192, 360]},
    "1": {"description": "白色郁金香", "mask": [213, 217, 341, 405]},
    "2": {"description": "木质花架", "mask": [363, 187, 480, 373]},
    "3": {"description": "视觉文本“春日花宴”（主标题）", "mask": [85, 43, 235, 103]},
    "4": {"description": "视觉文本“全场8折”（促销信息）", "mask": [300, 56, 427, 107]}
},  
{
    "prompt": "一碗热豆浆、一笼猪肉包子、一根油条，搭配视觉文本“元气早餐”和“现做现卖”，背景为米黄色。",
    "0": {"description": "热豆浆", "mask": [75, 202, 181, 347]},
    "1": {"description": "猪肉包子", "mask": [203, 231, 331, 389]},
    "2": {"description": "油条", "mask": [347, 217, 469, 360]},
    "3": {"description": "视觉文本“元气早餐”（主题）", "mask": [96, 50, 224, 93]},
    "4": {"description": "视觉文本“现做现卖”（特点）", "mask": [288, 56, 437, 101]}
},
{
    "prompt": "一条银质项链、一对珍珠耳环、一个编织手链，搭配视觉文本“手工市集”和“原创设计”，背景为浅灰色。",
    "0": {"description": "银质项链", "mask": [85, 187, 203, 331]},
    "1": {"description": "珍珠耳环", "mask": [224, 231, 320, 347]},
    "2": {"description": "编织手链", "mask": [341, 203, 453, 360]},
    "3": {"description": "视觉文本“手工市集”（活动名）", "mask": [107, 43, 245, 93]},
    "4": {"description": "视觉文本“原创设计”（亮点）", "mask": [300, 50, 427, 93]}
},
{
    "prompt": "一杯芒果沙冰、一杯柠檬水、一盘西瓜块，搭配视觉文本“夏日冰爽”和“第二杯半价”，背景为浅蓝色。",
    "0": {"description": "芒果沙冰", "mask": [64, 173, 192, 347]},
    "1": {"description": "柠檬水", "mask": [213, 217, 331, 373]},
    "2": {"description": "西瓜块", "mask": [347, 203, 480, 360]},
    "3": {"description": "视觉文本“夏日冰爽”（主题）", "mask": [96, 50, 235, 93]},
    "4": {"description": "视觉文本“第二杯半价”（活动）", "mask": [288, 56, 437, 101]}
},
{
    "prompt": "一幅生肖狗剪纸、一幅花鸟剪纸、一个剪纸相框，搭配视觉文本“非遗剪纸展”和“传承文化”，背景为米白色。",
    "0": {"description": "生肖狗剪纸", "mask": [70, 130, 180, 230]},
    "1": {"description": "花鸟剪纸", "mask": [200, 120, 320, 240]},
    "2": {"description": "剪纸相框", "mask": [340, 140, 440, 250]},
    "3": {"description": "视觉文本“非遗剪纸展”（展览名）", "mask": [80, 30, 240, 65]},
    "4": {"description": "视觉文本“传承文化”（理念）", "mask": [280, 35, 400, 65]}
},
{
    "prompt": "a blue beach towel, a pair of sunglasses, a yellow beach ball, with visual text \"Summer Beach Party\" and \"July 15th\" on a light blue background.",
    "0": {"description": "blue beach towel", "mask": [64, 261, 213, 417]},
    "1": {"description": "sunglasses", "mask": [235, 217, 320, 302]},
    "2": {"description": "yellow beach ball", "mask": [341, 231, 469, 390]},
    "3": {"description": "visual text \"Summer Beach Party\" (title)", "mask": [75, 50, 277, 101]},
    "4": {"description": "visual text \"July 15th\" (date)", "mask": [311, 56, 437, 101]}
},
{
    "prompt": "an old leather book, a vintage inkwell, a brass bookmark, with visual text \"Vintage Book Fair\" and \"Rare Books\" on a brown background.",
    "0": {"description": "old leather book", "mask": [75, 173, 224, 360]},
    "1": {"description": "vintage inkwell", "mask": [245, 231, 331, 331]},
    "2": {"description": "brass bookmark", "mask": [347, 203, 453, 347]},
    "3": {"description": "visual text \"Vintage Book Fair\" (event name)", "mask": [85, 43, 277, 93]},
    "4": {"description": "visual text \"Rare Books\" (feature)", "mask": [311, 50, 427, 93]}
},
{
    "prompt": "a lavender soap bar, a rose soap bar, a wooden soap dish, with visual text \"Handmade Soap Sale\" and \"Natural Ingredients\" on a white background.",
    "0": {"description": "lavender soap bar", "mask": [64, 203, 181, 347]},
    "1": {"description": "rose soap bar", "mask": [203, 217, 320, 360]},
    "2": {"description": "wooden soap dish", "mask": [341, 231, 469, 373]},
    "3": {"description": "visual text \"Handmade Soap Sale\" (title)", "mask": [75, 50, 277, 101]},
    "4": {"description": "visual text \"Natural Ingredients\" (advantage)", "mask": [300, 56, 448, 101]}
},
{
    "prompt": "a small white cat plush, a dog toy bone, a pet collar, with visual text \"Pet Adoption Day\" and \"Find Your Friend\" on a light gray background.",
    "0": {"description": "white cat plush", "mask": [75, 217, 192, 360]},
    "1": {"description": "dog toy bone", "mask": [213, 261, 331, 347]},
    "2": {"description": "pet collar", "mask": [347, 231, 453, 331]},
    "3": {"description": "visual text \"Pet Adoption Day\" (event title)", "mask": [75, 43, 299, 93]},
    "4": {"description": "visual text \"Find Your Friend\" (slogan)", "mask": [311, 50, 448, 93]}
},
{
    "prompt": "a pumpkin bread, a cinnamon roll, a cup of hot cocoa, with visual text \"Autumn Bakery\" and \"Warm Taste\" on a light orange background.",
    "0": {"description": "pumpkin bread", "mask": [64, 203, 192, 360]},
    "1": {"description": "cinnamon roll", "mask": [213, 231, 331, 373]},
    "2": {"description": "hot cocoa", "mask": [347, 217, 469, 360]},
    "3": {"description": "visual text \"Autumn Bakery\" (theme)", "mask": [85, 50, 245, 93]},
    "4": {"description": "visual text \"Warm Taste\" (slogan)", "mask": [288, 56, 427, 101]}
}

]