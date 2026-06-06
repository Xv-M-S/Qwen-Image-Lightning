

regional_prompt_mask_pairs = {
    "0": {
        "description": '''a chalkboard sign reading "Qwen Coffee 😊 $2 per cup"''',
        "mask": [128, 240, 384, 640]
    },
    "1": {
        "description": '''a neon light  displaying "通义千问"''',
        "mask": [500, 48, 840, 160]
    },
    "2": {
        "description": '''"π≈3.1415926-53589793-23846264-33832795-02384197" is written on the wall''',
        "mask": [500, 640, 756, 780]
    }
}

regional_prompt_mask_pairs2 = {
    "0": {
        "description": '''"咖啡店"''',
        "mask": [128, 240, 256, 640]
    },
    "1": {
        "description": '''"通义千问"''',
        "mask": [500, 48, 840, 160]
    },
    "2": {
        "description": '''"π≈3.1415926"''',
        "mask": [500, 640, 756, 780]
    }
}


# regional_prompt_mask_pairs_sample = {
#     "prompt": "一束粉色玫瑰、一盆白色郁金香、一个木质花架，搭配视觉文本"春日花宴"和"全场8折"，背景为浅粉色。",
#     "0": {"description": "粉色玫瑰", "mask": [64, 173, 192, 360]},
#     "1": {"description": "白色郁金香", "mask": [213, 217, 341, 405]},
#     "2": {"description": "木质花架", "mask": [363, 187, 480, 373]},
#     "3": {"description": "视觉文本"春日花宴"（主标题）", "mask": [85, 43, 235, 103]},
#     "4": {"description": "视觉文本"全场8折"（促销信息）", "mask": [300, 56, 427, 107]}
# }



regional_prompt_mask_pairs_sample = {
    "prompt": '纯黑色背景，一个绿色的视觉文本，在中部偏上的位置，内容为"视觉文本生成"，一个蓝色的视觉文本,内容为"Can you"，在中部偏下',
    "0": {"description": '"视觉文本生成"', "mask": [32, 32, 64, 224]},
    "1": {"description": "can you", "mask": [160, 64, 192, 224]},
}



# 整合后的大列表 - 可直接替换 testCase.py 中的 regional_prompt_mask_pairs2
regional_prompt_mask_pairs2_old = [
    {
        "prompt": 'A laptop is open in the center with a keyboard in front of it, and a TV screen can be seen in the background to the right.',
        "0": {"description": "tv", "mask": [0.711, 0.352, 0.992, 0.797]},
        "1": {"description": "laptop", "mask": [0.031, 0.477, 0.367, 0.695]},
        "2": {"description": "keyboard", "mask": [0.414, 0.633, 0.812, 0.742]},
    },
    {
        "prompt": 'The refrigerator is to the left, the oven is in the center, and the sink is to the right.',
        "0": {"description": "oven", "mask": [0.406, 0.320, 0.586, 0.875]},
        "1": {"description": "refrigerator", "mask": [0.023, 0.211, 0.219, 0.594]},
        "2": {"description": "sink", "mask": [0.773, 0.367, 0.969, 0.453]},
    },
    {
        "prompt": 'A person on the left is facing and holding the string of a kite to the right, while another person is standing in the background to the right, near another kite on the ground.',
        "0": {"description": "kite", "mask": [0.242, 0.305, 0.828, 0.461]},
        "1": {"description": "kite", "mask": [0.852, 0.172, 0.992, 0.336]},
        "2": {"description": "person", "mask": [0.258, 0.539, 0.352, 0.742]},
        "3": {"description": "person", "mask": [0.125, 0.531, 0.219, 0.742]},
    },
    {
        "prompt": 'A dining table is in the foreground with a chair to its left, an oven is seen in the background to the right, and a refrigerator stands to the left of the oven.',
        "0": {"description": "dining table", "mask": [0.000, 0.656, 0.461, 0.828]},
        "1": {"description": "refrigerator", "mask": [0.523, 0.391, 0.641, 0.758]},
        "2": {"description": "chair", "mask": [0.359, 0.609, 0.500, 0.773]},
        "3": {"description": "oven", "mask": [0.633, 0.531, 0.695, 0.711]},
    },
    {
        "prompt": 'A bus is parked on the left with people standing next to it, and there are skis and a backpack on the snow-covered ground to the right.',
        "0": {"description": "bus", "mask": [0.000, 0.211, 0.469, 0.484]},
        "1": {"description": "backpack", "mask": [0.094, 0.586, 0.336, 0.727]},
        "2": {"description": "person", "mask": [0.805, 0.297, 0.914, 0.523]},
        "3": {"description": "skis", "mask": [0.117, 0.742, 0.508, 0.805]},
    }
]



regional_prompt_mask_pairs2_back = [
    {
        "prompt": 'Create a poster for a wine tasting event featuring a serene vineyard scenery in the background. In the center of the poster, place a person savoring a glass of wine. Above the central figure, position the text "Wine Tasting Event" and below it, add "Exploring the Allure of Wine". At the bottom of the poster, include the date "October 1st".',
        "0": {"description": "Serene Vineyard Scenery", "mask": [0.5, 0.5, 1, 1]},
        "1": {"description": "Person Savoring a Glass of Wine", "mask": [0.5, 0.5, 0.3, 0.4]},
        "2": {"description": "Wine Tasting Event", "mask": [0.5, 0.4, 0.6, 0.1]},
        "3": {"description": "Exploring the Allure of Wine", "mask": [0.5, 0.6, 0.8, 0.1]},
        "4": {"description": "October 1st", "mask": [0.5, 0.9, 0.2, 0.1]},
    },
    {
        "prompt": 'Conceptualize an outdoor adventure-themed poster for a premium sportswear brand, set against a breathtaking snowy mountain landscape. In the foreground, a climber\'s silhouette is positioned on the lower right, ascending a steep, snow-covered slope. The text "Challenge Accepted Conquer Your Limits" is prominently displayed in bold, modern font at the top center of the poster, while the phrase "Hiking Climbing Exploring Repeat" is elegantly placed below it, slightly to the left, creating a balanced and dynamic visual flow.',
        "0": {"description": "Snowy Mountain Landscape", "mask": [0.5, 0.5, 1, 1]},
        "1": {"description": "Climber's Silhouette", "mask": [0.8, 0.2, 0.15, 0.3]},
        "2": {"description": "Challenge Accepted Conquer Your Limits", "mask": [0.5, 0.9, 0.8, 0.1]},
        "3": {"description": "Hiking Climbing Exploring Repeat", "mask": [0.4, 0.8, 0.6, 0.1]},
    },
    {
        "prompt": 'Create a Christmas greeting card featuring a cheerful snowman and a pile of gifts as the main characters. The snowman, adorned with a cozy scarf and a top hat, stands to the left of the gifts, which are neatly stacked on the right. Above the snowman, in elegant script, the words "Merry Christmas" are written. Below the snowman, the message "Wish you a Merry Christmas" is displayed. To the right of the gifts, the reminder "Pay attention to keeping warm during the holiday" is gently placed. At the bottom of the card, the final message "Send blessings to others" completes the festive scene.',
        "0": {"description": "Cheerful Snowman with Scarf and Top Hat", "mask": [0.3, 0.5, 0.2, 0.4]},
        "1": {"description": "Pile of Gifts", "mask": [0.7, 0.5, 0.2, 0.4]},
        "2": {"description": "Merry Christmas", "mask": [0.3, 0.7, 0.4, 0.1]},
        "3": {"description": "Wish you a Merry Christmas", "mask": [0.3, 0.4, 0.4, 0.1]},
        "4": {"description": "Pay attention to keeping warm during the holiday", "mask": [0.7, 0.4, 0.6, 0.1]},
        "5": {"description": "Send blessings to others", "mask": [0.5, 0.1, 0.4, 0.1]},
    },
    {
        "prompt": 'On a top light green background, a blooming cherry blossom tree stands prominently to the left, its delicate pink petals scattered gently around. To the right, a wicker basket is placed, partially filled with more cherry blossoms. In front of the basket, a glass of lemonade sits, glistening in the soft light. The visual text "Spring Garden" and "Fresh Bloom" are elegantly written above the scene, enhancing the serene and refreshing atmosphere.',
        "0": {"description": "Top Light Green Background", "mask": [0.5, 0.5, 1, 1]},
        "1": {"description": "Blooming Cherry Blossom Tree", "mask": [0.25, 0.5, 0.4, 0.8]},
        "2": {"description": "Scattered Pink Petals", "mask": [0.5, 0.6, 0.6, 0.2]},
        "3": {"description": "Wicker Basket", "mask": [0.75, 0.5, 0.2, 0.3]},
        "4": {"description": "Cherry Blossoms in Basket", "mask": [0.75, 0.5, 0.2, 0.2]},
        "5": {"description": "Glass of Lemonade", "mask": [0.75, 0.6, 0.1, 0.2]},
        "6": {"description": "Spring Garden", "mask": [0.5, 0.9, 0.4, 0.1]},
        "7": {"description": "Fresh Bloom", "mask": [0.5, 0.85, 0.4, 0.1]},
    },
    {
        "prompt": '晨光中的江南水乡河道，中央水面停泊着一艘彩绘龙舟。左岸的石阶上放着一篮青绿粽子，右侧的木门框上插着带露艾草。碧水与白墙相映成趣，尽显端午民俗氛围。在龙舟正上方悬浮着墨绿粗体大字"端午"，左侧配有刻有"五月初五"的龙纹徽章，而左上角则有一个粽子图标。画面底部中央是白色小字"驱邪纳吉，岁岁安康"。',
        "0": {"description": "江南水乡河道", "mask": [0.5, 0.5, 1, 1]},
        "1": {"description": "彩绘龙舟", "mask": [0.5, 0.5, 0.4, 0.2]},
        "2": {"description": "一篮青绿粽子", "mask": [0.3, 0.6, 0.1, 0.1]},
        "3": {"description": "带露艾草", "mask": [0.7, 0.6, 0.1, 0.1]},
        "4": {"description": "端午", "mask": [0.5, 0.4, 0.2, 0.1]},
        "5": {"description": "五月初五", "mask": [0.4, 0.4, 0.1, 0.1]},
        "6": {"description": "粽子图标", "mask": [0.2, 0.8, 0.1, 0.1]},
        "7": {"description": "驱邪纳吉，岁岁安康", "mask": [0.5, 0.9, 0.4, 0.1]},
    },
    {
        "prompt": '深秋的森林空地铺满了金黄的落叶，营造出一片宁静而自然的氛围。左前方，一只尾巴蓬松的红狐静静地站立着，与中右侧那棵火红的枫树遥相呼应。右后方卧着一块覆有苔藓的青灰色巨石，增添了几分古朴的气息。在枫树前方，悬浮着深褐色手写体文字"林间有信"，左上角配有枫叶图标，右上角则是一个狐狸剪影小标。文字右侧是带有"秋藏"篆字的圆形徽章，画面底部中央则用小号白色字体写着副标题"万物有时"。整个画面布局和谐，充满了秋天的韵味。',
        "0": {"description": "金黄落叶", "mask": [0.5, 0.5, 1, 1]},
        "1": {"description": "红狐", "mask": [0.25, 0.4, 0.15, 0.2]},
        "2": {"description": "火红枫树", "mask": [0.75, 0.5, 0.2, 0.4]},
        "3": {"description": "青灰色巨石", "mask": [0.8, 0.6, 0.15, 0.2]},
        "4": {"description": "林间有信", "mask": [0.7, 0.3, 0.2, 0.1]},
        "5": {"description": "枫叶图标", "mask": [0.1, 0.1, 0.05, 0.05]},
        "6": {"description": "狐狸剪影小标", "mask": [0.9, 0.1, 0.05, 0.05]},
        "7": {"description": "秋藏篆字圆形徽章", "mask": [0.8, 0.3, 0.1, 0.1]},
        "8": {"description": "万物有时", "mask": [0.5, 0.9, 0.2, 0.05]},
    },
    {
        "prompt": 'A cinematic movie poster for a film titled \'Solitary Journeys\', starring Elara Voss. The background features a vast, desolate wasteland, with a lone, small figure walking toward a mysterious, unrecognizable form of transportation—perhaps a derelict bus or an abstract structure—situated in the distance. Captured through a wide-angle lens, the scene evokes a sense of being lost, helplessness, and desolation. The color palette is muted and dusty, dominated by greys, ochres, and faded blues, enhancing the melancholic and introspective mood. At the top, the title \'Solitary Journeys\' is displayed in a bold, distressed sans-serif font. Below the title, the subtitle \'Elara Voss\' and the tagline \'WANDERING THROUGH THE UNKNOWN\' are positioned in a smaller, elegant typeface, adding to the overall emotional weight of isolation.',
        "0": {"description": "Vast Desolate Wasteland", "mask": [0.5, 0.5, 1, 1]},
        "1": {"description": "Lone Small Figure", "mask": [0.5, 0.4, 0.05, 0.1]},
        "2": {"description": "Mysterious Transportation", "mask": [0.5, 0.3, 0.2, 0.1]},
        "3": {"description": "Solitary Journeys", "mask": [0.5, 0.9, 0.8, 0.1]},
        "4": {"description": "Elara Voss", "mask": [0.5, 0.8, 0.6, 0.05]},
        "5": {"description": "WANDERING THROUGH THE UNKNOWN", "mask": [0.5, 0.75, 0.8, 0.05]},
    },
    {
        "prompt": 'An autumn farmstead at golden hour, bathed in warm amber light, features a cornucopia full of pumpkins and apples sitting on the left wooden porch. To the right, wild turkeys walk through dry cornfields, while a red barn with a smoking chimney stands in the distance. In the center, bold burnt orange serif text "THANKSGIVING" floats prominently, with a feather-and-wheat emblem just below it. At the top right corner, a maple leaf icon adds a touch of seasonal charm, and at the bottom, a soft brown subtitle "Give thanks, share warmth." is centered, completing the serene and inviting scene.',
        "0": {"description": "Autumn Farmstead", "mask": [0.5, 0.5, 1, 1]},
        "1": {"description": "Cornucopia with Pumpkins and Apples", "mask": [0.2, 0.4, 0.3, 0.3]},
        "2": {"description": "Wild Turkeys in Dry Cornfields", "mask": [0.8, 0.3, 0.2, 0.2]},
        "3": {"description": "Red Barn with Smoking Chimney", "mask": [0.7, 0.6, 0.2, 0.3]},
        "4": {"description": "THANKSGIVING", "mask": [0.5, 0.5, 0.6, 0.1]},
        "5": {"description": "Feather-and-Wheat Emblem", "mask": [0.5, 0.55, 0.2, 0.1]},
        "6": {"description": "Maple Leaf Icon", "mask": [0.9, 0.1, 0.1, 0.1]},
        "7": {"description": "Give thanks, share warmth.", "mask": [0.5, 0.9, 0.6, 0.1]},
    }
]


regional_prompt_mask_pairs2_old = [
    {
        "prompt": 'Create a poster for a wine tasting event featuring a serene vineyard scenery in the background. In the center of the poster, place a person savoring a glass of wine. Above the central figure, position the text "Wine Tasting Event" and below it, add "Exploring the Allure of Wine". At the bottom of the poster, include the date "October 1st".',
        "0": {"description": "serene Vineyard Scenery", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "person Savoring a Glass of Wine", "mask": [0.35, 0.35, 0.65, 0.75]},
        "2": {"description": "Wine Tasting Event", "mask": [0.1, 0.1, 0.9, 0.25]},
        "3": {"description": "Exploring the Allure of Wine", "mask": [0.15, 0.78, 0.85, 0.9]},
        "4": {"description": "October 1st", "mask": [0.35, 0.92, 0.65, 0.98]},
    },
    {
        "prompt": 'Conceptualize an outdoor adventure-themed poster for a premium sportswear brand, set against a breathtaking snowy mountain landscape. In the foreground, a climber\'s silhouette is positioned on the lower right, ascending a steep, snow-covered slope. The text "Challenge Accepted Conquer Your Limits" is prominently displayed in bold, modern font at the top center of the poster, while the phrase "Hiking Climbing Exploring Repeat" is elegantly placed below it, slightly to the left, creating a balanced and dynamic visual flow.',
        "0": {"description": "snowy Mountain Landscape", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "climber\'s Silhouette", "mask": [0.65, 0.6, 0.95, 0.95]},
        "2": {"description": "Challenge Accepted Conquer Your Limits", "mask": [0.1, 0.05, 0.9, 0.2]},
        "3": {"description": "Hiking Climbing Exploring Repeat", "mask": [0.15, 0.22, 0.75, 0.32]},
    },
    {
        "prompt": 'Create a Christmas greeting card featuring a cheerful snowman and a pile of gifts as the main characters. The snowman, adorned with a cozy scarf and a top hat, stands to the left of the gifts, which are neatly stacked on the right. Above the snowman, in elegant script, the words "Merry Christmas" are written. Below the snowman, the message "Wish you a Merry Christmas" is displayed. To the right of the gifts, the reminder "Pay attention to keeping warm during the holiday" is gently placed. At the bottom of the card, the final message "Send blessings to others" completes the festive scene.',
        "0": {"description": "cheerful Snowman with Scarf and Top Hat", "mask": [0.1, 0.3, 0.45, 0.8]},
        "1": {"description": "Pile of Gifts", "mask": [0.55, 0.4, 0.9, 0.8]},
        "2": {"description": "Merry Christmas", "mask": [0.1, 0.1, 0.5, 0.25]},
        "3": {"description": "Wish you a Merry Christmas", "mask": [0.1, 0.82, 0.5, 0.9]},
        "4": {"description": "Pay attention to keeping warm during the holiday", "mask": [0.55, 0.85, 0.95, 0.95]},
        "5": {"description": "Send blessings to others", "mask": [0.2, 0.92, 0.8, 0.98]},
    },
    {
        "prompt": 'On a top light green background, a blooming cherry blossom tree stands prominently to the left, its delicate pink petals scattered gently around. To the right, a wicker basket is placed, partially filled with more cherry blossoms. In front of the basket, a glass of lemonade sits, glistening in the soft light. The visual text "Spring Garden" and "Fresh Bloom" are elegantly written above the scene, enhancing the serene and refreshing atmosphere.',
        "0": {"description": "Top Light Green Background", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "Blooming Cherry Blossom Tree", "mask": [0.0, 0.1, 0.45, 0.9]},
        "2": {"description": "Scattered Pink Petals", "mask": [0.4, 0.4, 0.7, 0.7]},
        "3": {"description": "Wicker Basket", "mask": [0.6, 0.5, 0.9, 0.8]},
        "4": {"description": "Cherry Blossoms in Basket", "mask": [0.62, 0.52, 0.88, 0.7]},
        "5": {"description": "Glass of Lemonade", "mask": [0.65, 0.72, 0.85, 0.88]},
        "6": {"description": "Spring Garden", "mask": [0.2, 0.05, 0.8, 0.2]},
        "7": {"description": "Fresh Bloom", "mask": [0.3, 0.22, 0.7, 0.32]},
    },
    {
        "prompt": '晨光中的江南水乡河道，中央水面停泊着一艘彩绘龙舟。左岸的石阶上放着一篮青绿粽子，右侧的木门框上插着带露艾草。碧水与白墙相映成趣，尽显端午民俗氛围。在龙舟正上方悬浮着墨绿粗体大字"端午"，左侧配有刻有"五月初五"的龙纹徽章，而左上角则有一个粽子图标。画面底部中央是白色小字"驱邪纳吉，岁岁安康"。',
        "0": {"description": "江南水乡河道", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "彩绘龙舟", "mask": [0.2, 0.4, 0.8, 0.7]},
        "2": {"description": "一篮青绿粽子", "mask": [0.05, 0.5, 0.25, 0.7]},
        "3": {"description": "带露艾草", "mask": [0.75, 0.3, 0.95, 0.5]},
        "4": {"description": "端午", "mask": [0.3, 0.1, 0.7, 0.25]},
        "5": {"description": "五月初五", "mask": [0.05, 0.15, 0.25, 0.25]},
        "6": {"description": "粽子图标", "mask": [0.05, 0.05, 0.15, 0.15]},
        "7": {"description": "驱邪纳吉，岁岁安康", "mask": [0.2, 0.9, 0.8, 0.98]},
    },
    {
        "prompt": '深秋的森林空地铺满了金黄的落叶，营造出一片宁静而自然的氛围。左前方，一只尾巴蓬松的红狐静静地站立着，与中右侧那棵火红的枫树遥相呼应。右后方卧着一块覆有苔藓的青灰色巨石，增添了几分古朴的气息。在枫树前方，悬浮着深褐色手写体文字"林间有信"，左上角配有枫叶图标，右上角则是一个狐狸剪影小标。文字右侧是带有"秋藏"篆字的圆形徽章，画面底部中央则用小号白色字体写着副标题"万物有时"。整个画面布局和谐，充满了秋天的韵味。',
        "0": {"description": "金黄落叶", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "红狐", "mask": [0.05, 0.4, 0.35, 0.8]},
        "2": {"description": "火红枫树", "mask": [0.6, 0.2, 0.95, 0.8]},
        "3": {"description": "青灰色巨石", "mask": [0.7, 0.6, 0.95, 0.85]},
        "4": {"description": "林间有信", "mask": [0.1, 0.15, 0.55, 0.3]},
        "5": {"description": "枫叶图标", "mask": [0.05, 0.05, 0.15, 0.15]},
        "6": {"description": "狐狸剪影小标", "mask": [0.85, 0.05, 0.95, 0.15]},
        "7": {"description": "秋藏篆字圆形徽章", "mask": [0.6, 0.35, 0.75, 0.5]},
        "8": {"description": "万物有时", "mask": [0.35, 0.92, 0.65, 0.98]},
    },
    {
        "prompt": 'A cinematic movie poster for a film titled \'Solitary Journeys\', starring Elara Voss. The background features a vast, desolate wasteland, with a lone, small figure walking toward a mysterious, unrecognizable form of transportation—perhaps a derelict bus or an abstract structure—situated in the distance. Captured through a wide-angle lens, the scene evokes a sense of being lost, helplessness, and desolation. The color palette is muted and dusty, dominated by greys, ochres, and faded blues, enhancing the melancholic and introspective mood. At the top, the title \'Solitary Journeys\' is displayed in a bold, distressed sans-serif font. Below the title, the subtitle \'Elara Voss\' and the tagline \'WANDERING THROUGH THE UNKNOWN\' are positioned in a smaller, elegant typeface, adding to the overall emotional weight of isolation.',
        "0": {"description": "Vast Desolate Wasteland", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "Lone Small Figure", "mask": [0.45, 0.6, 0.55, 0.75]},
        "2": {"description": "Mysterious Transportation", "mask": [0.3, 0.3, 0.7, 0.5]},
        "3": {"description": "Solitary Journeys", "mask": [0.1, 0.05, 0.9, 0.2]},
        "4": {"description": "Elara Voss", "mask": [0.3, 0.22, 0.7, 0.28]},
        "5": {"description": "WANDERING THROUGH THE UNKNOWN", "mask": [0.2, 0.3, 0.8, 0.35]},
    },
    {
        "prompt": 'An autumn farmstead at golden hour, bathed in warm amber light, features a cornucopia full of pumpkins and apples sitting on the left wooden porch. To the right, wild turkeys walk through dry cornfields, while a red barn with a smoking chimney stands in the distance. In the center, bold burnt orange serif text "THANKSGIVING" floats prominently, with a feather-and-wheat emblem just below it. At the top right corner, a maple leaf icon adds a touch of seasonal charm, and at the bottom, a soft brown subtitle "Give thanks, share warmth." is centered, completing the serene and inviting scene.',
        "0": {"description": "Autumn Farmstead", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "Cornucopia with Pumpkins and Apples", "mask": [0.05, 0.5, 0.4, 0.85]},
        "2": {"description": "Wild Turkeys in Dry Cornfields", "mask": [0.6, 0.5, 0.9, 0.8]},
        "3": {"description": "Red Barn with Smoking Chimney", "mask": [0.65, 0.2, 0.95, 0.5]},
        "4": {"description": "THANKSGIVING", "mask": [0.1, 0.2, 0.9, 0.35]},
        "5": {"description": "Feather-and-Wheat Emblem", "mask": [0.35, 0.38, 0.65, 0.48]},
        "6": {"description": "Maple Leaf Icon", "mask": [0.85, 0.05, 0.95, 0.15]},
        "7": {"description": "Give thanks, share warmth.", "mask": [0.2, 0.9, 0.8, 0.98]},
    }
]


regional_prompt_mask_pairs2 = [
    {
        "prompt": 'Create a poster for a wine tasting event featuring a serene vineyard scenery in the background. In the center of the poster, place a person savoring a glass of wine. Above the central figure, position the text "Wine Tasting Event" and below it, add "Exploring the Allure of Wine". At the bottom of the poster, include the date "October 1st".',
        "0": {"description": "serene vineyard scenery", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "a person savoring a glass of wine", "mask": [0.35, 0.35, 0.65, 0.75]},
        "2": {"description": "Wine Tasting Event", "mask": [0.1, 0.1, 0.9, 0.25]},
        "3": {"description": "Exploring the Allure of Wine", "mask": [0.15, 0.78, 0.85, 0.9]},
        "4": {"description": "October 1st", "mask": [0.35, 0.92, 0.65, 0.98]},
    },
    {
        "prompt": 'Conceptualize an outdoor adventure-themed poster for a premium sportswear brand, set against a breathtaking snowy mountain landscape. In the foreground, a climber\'s silhouette is positioned on the lower right, ascending a steep, snow-covered slope. The text "Challenge Accepted Conquer Your Limits" is prominently displayed in bold, modern font at the top center of the poster, while the phrase "Hiking Climbing Exploring Repeat" is elegantly placed below it, slightly to the left, creating a balanced and dynamic visual flow.',
        "0": {"description": "snowy mountain landscape", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "a climber\'s silhouette", "mask": [0.65, 0.6, 0.95, 0.95]},
        "2": {"description": "Challenge Accepted Conquer Your Limits", "mask": [0.1, 0.05, 0.9, 0.2]},
        "3": {"description": "Hiking Climbing Exploring Repeat", "mask": [0.15, 0.22, 0.75, 0.32]},
    },
    {
        "prompt": 'Create a Christmas greeting card featuring a cheerful snowman and a pile of gifts as the main characters. The snowman, adorned with a cozy scarf and a top hat, stands to the left of the gifts, which are neatly stacked on the right. Above the snowman, in elegant script, the words "Merry Christmas" are written. Below the snowman, the message "Wish you a Merry Christmas" is displayed. To the right of the gifts, the reminder "Pay attention to keeping warm during the holiday" is gently placed. At the bottom of the card, the final message "Send blessings to others" completes the festive scene.',
        "0": {"description": "a cheerful snowman", "mask": [0.1, 0.3, 0.45, 0.8]},
        "1": {"description": "a pile of gifts", "mask": [0.55, 0.4, 0.9, 0.8]},
        "2": {"description": "Merry Christmas", "mask": [0.1, 0.1, 0.5, 0.25]},
        "3": {"description": "Wish you a Merry Christmas", "mask": [0.1, 0.82, 0.5, 0.9]},
        "4": {"description": "Pay attention to keeping warm during the holiday", "mask": [0.55, 0.85, 0.95, 0.95]},
        "5": {"description": "Send blessings to others", "mask": [0.2, 0.92, 0.8, 0.98]},
    },
    {
        "prompt": 'On a top light green background, a blooming cherry blossom tree stands prominently to the left, its delicate pink petals scattered gently around. To the right, a wicker basket is placed, partially filled with more cherry blossoms. In front of the basket, a glass of lemonade sits, glistening in the soft light. The visual text "Spring Garden" and "Fresh Bloom" are elegantly written above the scene, enhancing the serene and refreshing atmosphere.',
        "0": {"description": "top light green background", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "a blooming cherry blossom tree", "mask": [0.0, 0.1, 0.45, 0.9]},
        "2": {"description": "scattered pink petals", "mask": [0.4, 0.4, 0.7, 0.7]},
        "3": {"description": "a wicker basket", "mask": [0.6, 0.5, 0.9, 0.8]},
        "4": {"description": "cherry blossoms in basket", "mask": [0.62, 0.52, 0.88, 0.7]},
        "5": {"description": "a glass of lemonade", "mask": [0.65, 0.72, 0.85, 0.88]},
        "6": {"description": "Spring Garden", "mask": [0.2, 0.05, 0.8, 0.2]},
        "7": {"description": "Fresh Bloom", "mask": [0.3, 0.22, 0.7, 0.32]},
    },
    {
        "prompt": '晨光中的江南水乡河道，中央水面停泊着一艘彩绘龙舟。左岸的石阶上放着一篮青绿粽子，右侧的木门框上插着带露艾草。碧水与白墙相映成趣，尽显端午民俗氛围。在龙舟正上方悬浮着墨绿粗体大字"端午"，左侧配有刻有"五月初五"的龙纹徽章，而左上角则有一个粽子图标。画面底部中央是白色小字"驱邪纳吉，岁岁安康"。',
        "0": {"description": "江南水乡河道", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "一艘彩绘龙舟", "mask": [0.2, 0.4, 0.8, 0.7]},
        "2": {"description": "一篮青绿粽子", "mask": [0.05, 0.5, 0.25, 0.7]},
        "3": {"description": "带露艾草", "mask": [0.75, 0.3, 0.95, 0.5]},
        "4": {"description": "端午", "mask": [0.3, 0.1, 0.7, 0.25]},
        "5": {"description": "五月初五", "mask": [0.05, 0.15, 0.25, 0.25]},
        "6": {"description": "粽子图标", "mask": [0.05, 0.05, 0.15, 0.15]},
        "7": {"description": "驱邪纳吉，岁岁安康", "mask": [0.2, 0.9, 0.8, 0.98]},
    },
    {
        "prompt": '深秋的森林空地铺满了金黄的落叶，营造出一片宁静而自然的氛围。左前方，一只尾巴蓬松的红狐静静地站立着，与中右侧那棵火红的枫树遥相呼应。右后方卧着一块覆有苔藓的青灰色巨石，增添了几分古朴的气息。在枫树前方，悬浮着深褐色手写体文字"林间有信"，左上角配有枫叶图标，右上角则是一个狐狸剪影小标。文字右侧是带有"秋藏"篆字的圆形徽章，画面底部中央则用小号白色字体写着副标题"万物有时"。整个画面布局和谐，充满了秋天的韵味。',
        "0": {"description": "金黄的落叶", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "一只尾巴蓬松的红狐", "mask": [0.05, 0.4, 0.35, 0.8]},
        "2": {"description": "一棵火红的枫树", "mask": [0.6, 0.2, 0.95, 0.8]},
        "3": {"description": "一块覆有苔藓的青灰色巨石", "mask": [0.7, 0.6, 0.95, 0.85]},
        "4": {"description": "林间有信", "mask": [0.1, 0.15, 0.55, 0.3]},
        "5": {"description": "枫叶图标", "mask": [0.05, 0.05, 0.15, 0.15]},
        "6": {"description": "狐狸剪影小标", "mask": [0.85, 0.05, 0.95, 0.15]},
        "7": {"description": "秋藏篆字的圆形徽章", "mask": [0.6, 0.35, 0.75, 0.5]},
        "8": {"description": "万物有时", "mask": [0.35, 0.92, 0.65, 0.98]},
    },
    {
        "prompt": 'A cinematic movie poster for a film titled \'Solitary Journeys\', starring Elara Voss. The background features a vast, desolate wasteland, with a lone, small figure walking toward a mysterious, unrecognizable form of transportation—perhaps a derelict bus or an abstract structure—situated in the distance. Captured through a wide-angle lens, the scene evokes a sense of being lost, helplessness, and desolation. The color palette is muted and dusty, dominated by greys, ochres, and faded blues, enhancing the melancholic and introspective mood. At the top, the title \'Solitary Journeys\' is displayed in a bold, distressed sans-serif font. Below the title, the subtitle \'Elara Voss\' and the tagline \'WANDERING THROUGH THE UNKNOWN\' are positioned in a smaller, elegant typeface, adding to the overall emotional weight of isolation.',
        "0": {"description": "vast, desolate wasteland", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "a lone, small figure", "mask": [0.45, 0.6, 0.55, 0.75]},
        "2": {"description": "a mysterious, unrecognizable form of transportation", "mask": [0.3, 0.3, 0.7, 0.5]},
        "3": {"description": "Solitary Journeys", "mask": [0.1, 0.05, 0.9, 0.2]},
        "4": {"description": "Elara Voss", "mask": [0.3, 0.22, 0.7, 0.28]},
        "5": {"description": "WANDERING THROUGH THE UNKNOWN", "mask": [0.2, 0.3, 0.8, 0.35]},
    },
    {
        "prompt": 'An autumn farmstead at golden hour, bathed in warm amber light, features a cornucopia full of pumpkins and apples sitting on the left wooden porch. To the right, wild turkeys walk through dry cornfields, while a red barn with a smoking chimney stands in the distance. In the center, bold burnt orange serif text "THANKSGIVING" floats prominently, with a feather-and-wheat emblem just below it. At the top right corner, a maple leaf icon adds a touch of seasonal charm, and at the bottom, a soft brown subtitle "Give thanks, share warmth." is centered, completing the serene and inviting scene.',
        "0": {"description": "autumn farmstead", "mask": [0.0, 0.0, 1.0, 1.0]},
        "1": {"description": "a cornucopia full of pumpkins and apples", "mask": [0.05, 0.5, 0.4, 0.85]},
        "2": {"description": "wild turkeys", "mask": [0.6, 0.5, 0.9, 0.8]},
        "3": {"description": "a red barn with a smoking chimney", "mask": [0.65, 0.2, 0.95, 0.5]},
        "4": {"description": "THANKSGIVING", "mask": [0.1, 0.2, 0.9, 0.35]},
        "5": {"description": "a feather-and-wheat emblem", "mask": [0.35, 0.38, 0.65, 0.48]},
        "6": {"description": "maple leaf icon", "mask": [0.85, 0.05, 0.95, 0.15]},
        "7": {"description": "Give thanks, share warmth.", "mask": [0.2, 0.9, 0.8, 0.98]},
    }
]