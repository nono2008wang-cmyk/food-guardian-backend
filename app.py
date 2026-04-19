import cv2
import numpy as np
import base64
from ultralytics import YOLO
import math
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- 1. 擴充版食材清單 (YOLO 英文標籤) ---
custom_food_list = [
    # 蔬菜類 (Leafy & Root Vegetables)
    "Cabbage", "Chinese Cabbage", "Bok Choy", "Spinach", "Water Spinach", "Sweet Potato Leaves", "A-Choy", "Basil", "Coriander", "Celery",
    "Green Onion", "Whole Garlic Bulb", "Ginger", "Chili", "Onion", "Green Pepper", "Bell Pepper", "Cucumber", "Sponge Gourd", 
    "Bitter Gourd", "Winter Melon", "Pumpkin", "Zucchini", "Eggplant", "Tomato", "Okra", "Corn", "Baby Corn", "Lotus Root", "Burdock",
    "Carrot", "White Radish", "Potato", "Sweet Potato", "Taro", "Bamboo Shoot", "Water Bamboo", "Asparagus", "Leek", "Chives",
    "Bean Sprout", "Green Bean", "Edamame", "Pea", "Shiitake", "Enoki Mushroom", "King Oyster Mushroom", "Wood Ear Mushroom", "Broccoli", "Cauliflower",
    
    # 水果類 (Fruits)
    "Red Apple", "Green Apple", "Banana", "Pineapple", "Papaya", "Watermelon", "Melon", "Grape", "Strawberry", "Orange", "Tangerine", 
    "Lemon", "Lime", "Passion Fruit", "Mango", "Guava", "Pear", "Dragon Fruit", "Kiwi", "Peach", "Cherry", "Blueberry", "Avocado", "Grapefruit", "Plum",
    
    # 肉類 (Meats - 多重描述增加命中率)
    "Raw pork meat", "Packaged raw pork", "Raw pork belly slice", "Raw pork chop", "Minced pork", "Pork ribs", "Pork knuckle",
    "Sausage", "Bacon", "Ham", "Hot dog", "Spam", "Pork floss",
    "Raw beef meat", "Raw beef steak", "Sliced raw beef", "Packaged raw beef", "Meat", "Beef slice", "Red meat", "Wagyu beef", "Beef tendon",
    "Raw chicken meat", "Raw chicken breast", "Raw chicken leg", "Raw chicken wing", "Whole chicken", "Chicken nugget", "Roast chicken",
    "Duck meat", "Lamb meat", "Mutton slice",
    
    # 海鮮類 (Seafood)
    "Raw salmon fillet", "Raw tuna fillet", "Raw cod fish", "Raw mackerel", "Raw tilapia fish", "Milkfish", "Grouper", "Snapper",
    "Raw shrimp", "Peeled shrimp", "Lobster", "Crab", "Raw clam", "Raw oyster", "Squid", "Octopus", "Cuttlefish", "Scallop", "Abalone", "Seaweed", "Kelp",
    
    # 主食與豆製品 (Staples & Soy)
    "Rice", "Brown rice", "Sticky rice", "Noodle", "Pasta", "Udon noodle", "Ramen", "Instant noodle", "Vermicelli", "Macaroni",
    "Toast", "Bread", "Bun", "Dumpling", "Wonton", "Steamed bun", "Tangyuan",
    "Egg", "Quail egg", "Duck egg", "Tofu", "Dried Tofu", "Bean Curd Skin", "Fried tofu", "Stinky tofu",
    
    # 奶製品與調味品 (Dairy & Seasonings)
    "Milk", "Soy Milk", "Oat Milk", "Yogurt", "Cheese", "Butter", "Margarine", "Cream", "Condensed milk",
    "Soy Sauce", "Thick soy sauce", "Ketchup", "Kimchi", "Vinegar", "Black vinegar", "Sesame Paste", "Chili sauce", "Mayonnaise",
    "Flour", "Sugar", "Brown sugar", "Salt", "Yeast", "Baking Powder", "Vanilla Extract", "Chocolate", "Whipping Cream", "Cream Cheese", "Matcha Powder",
    "Sesame Oil", "Oyster Sauce", "Rice Wine", "Cooking wine", "Star Anise", "Cinnamon", "Curry powder", "Pepper powder", "Black pepper",
    "Dried Shrimp", "Fermented Black Beans", "Doubanjiang", "Miso", "Shacha sauce",
    
    # 罐頭與飲料 (Canned & Drinks)
    "Tuna can", "Kimchi can", "Corn can", "Tomato can", "Pickled cucumber",
    "Coke", "Pepsi", "Sprite", "Milk tea", "Oolong tea", "Green tea", "Black tea", "Coffee", "Juice", "Beer", "Wine", "Mineral water"
]

# --- 2. 食材英翻中字典 (多對一映射) ---
translation_dict = {
    # 蔬菜類
    "Cabbage": "高麗菜", "Chinese Cabbage": "大白菜", "Bok Choy": "小白菜", "Spinach": "菠菜", "Water Spinach": "空心菜", "Sweet Potato Leaves": "地瓜葉", "A-Choy": "A菜", "Basil": "九層塔", "Coriander": "香菜", "Celery": "芹菜",
    "Green Onion": "青蔥", "Whole Garlic Bulb": "蒜頭", "Ginger": "薑", "Chili": "辣椒", "Onion": "洋蔥", "Green Pepper": "青椒", "Bell Pepper": "甜椒", "Cucumber": "小黃瓜", "Sponge Gourd": "絲瓜", "Bitter Gourd": "苦瓜", "Winter Melon": "冬瓜", "Pumpkin": "南瓜", "Zucchini": "櫛瓜", "Eggplant": "茄子", "Tomato": "番茄", "Okra": "秋葵", "Corn": "玉米", "Baby Corn": "玉米筍", "Lotus Root": "蓮藕", "Burdock": "牛蒡",
    "Carrot": "紅蘿蔔", "White Radish": "白蘿蔔", "Potato": "馬鈴薯", "Sweet Potato": "地瓜", "Taro": "芋頭", "Bamboo Shoot": "竹筍", "Water Bamboo": "筊白筍", "Asparagus": "蘆筍", "Leek": "蒜苗", "Chives": "韭菜",
    "Bean Sprout": "豆芽菜", "Green Bean": "四季豆", "Edamame": "毛豆", "Pea": "豌豆", "Shiitake": "香菇", "Enoki Mushroom": "金針菇", "King Oyster Mushroom": "杏鮑菇", "Wood Ear Mushroom": "木耳", "Broccoli": "青花菜", "Cauliflower": "白花椰菜",
    
    # 水果類
    "Red Apple": "蘋果", "Green Apple": "青蘋果", "Banana": "香蕉", "Pineapple": "鳳梨", "Papaya": "木瓜", "Watermelon": "西瓜", "Melon": "哈密瓜", "Grape": "葡萄", "Strawberry": "草莓", "Orange": "柳丁", "Tangerine": "橘子", "Lemon": "檸檬", "Lime": "萊姆", "Passion Fruit": "百香果", "Mango": "芒果", "Guava": "芭樂", "Pear": "水梨", "Dragon Fruit": "火龍果", "Kiwi": "奇異果", "Peach": "桃子", "Cherry": "櫻桃", "Blueberry": "藍莓", "Avocado": "酪梨", "Grapefruit": "葡萄柚", "Plum": "李子",
    
    # 肉類
    "Raw pork meat": "豬肉", "Packaged raw pork": "豬肉", "Raw pork belly slice": "豬五花", "Raw pork chop": "豬排", "Minced pork": "豬絞肉", "Pork ribs": "排骨", "Pork knuckle": "豬腳",
    "Sausage": "香腸", "Bacon": "培根", "Ham": "火腿", "Hot dog": "熱狗", "Spam": "午餐肉", "Pork floss": "肉鬆",
    "Raw beef meat": "牛肉", "Sliced raw beef": "牛肉", "Packaged raw beef": "牛肉", "Meat": "肉", "Beef slice": "牛肉片", "Red meat": "紅肉", "Raw beef steak": "牛排", "Wagyu beef": "和牛", "Beef tendon": "牛筋",
    "Raw chicken meat": "雞肉", "Raw chicken breast": "雞胸肉", "Raw chicken leg": "雞腿", "Raw chicken wing": "雞翅", "Whole chicken": "全雞", "Chicken nugget": "雞塊", "Roast chicken": "烤雞",
    "Duck meat": "鴨肉", "Lamb meat": "羊肉", "Mutton slice": "羊肉片",
    
    # 海鮮類
    "Raw salmon fillet": "鮭魚", "Raw tuna fillet": "鮪魚", "Raw cod fish": "鱈魚", "Raw mackerel": "鯖魚", "Raw tilapia fish": "吳郭魚", "Milkfish": "虱目魚", "Grouper": "石斑魚", "Snapper": "鯛魚",
    "Raw shrimp": "蝦子", "Peeled shrimp": "蝦仁", "Lobster": "龍蝦", "Crab": "螃蟹", "Raw clam": "蛤蜊", "Raw oyster": "牡蠣", "Squid": "魷魚", "Octopus": "章魚", "Cuttlefish": "花枝", "Scallop": "干貝", "Abalone": "鮑魚", "Seaweed": "紫菜", "Kelp": "海帶",
    
    # 主食與豆製品
    "Rice": "白米", "Brown rice": "糙米", "Sticky rice": "糯米", "Noodle": "麵條", "Pasta": "義大利麵", "Udon noodle": "烏龍麵", "Ramen": "拉麵", "Instant noodle": "泡麵", "Vermicelli": "冬粉", "Macaroni": "通心粉",
    "Toast": "吐司", "Bread": "麵包", "Bun": "餐包", "Dumpling": "水餃", "Wonton": "餛飩", "Steamed bun": "饅頭", "Tangyuan": "湯圓",
    "Egg": "雞蛋", "Quail egg": "鳥蛋", "Duck egg": "鴨蛋", "Tofu": "豆腐", "Dried Tofu": "豆乾", "Bean Curd Skin": "豆皮", "Fried tofu": "油豆腐", "Stinky tofu": "臭豆腐",
    
    # 奶製品與調味品
    "Milk": "鮮奶", "Soy Milk": "豆漿", "Oat Milk": "燕麥奶", "Yogurt": "優格", "Cheese": "起司", "Butter": "奶油", "Margarine": "乳瑪琳", "Cream": "鮮奶油", "Condensed milk": "煉乳",
    "Soy Sauce": "醬油", "Thick soy sauce": "醬油膏", "Ketchup": "番茄醬", "Kimchi": "泡菜", "Vinegar": "白醋", "Black vinegar": "烏醋", "Sesame Paste": "芝麻醬", "Chili sauce": "辣椒醬", "Mayonnaise": "美乃滋",
    "Flour": "麵粉", "Sugar": "砂糖", "Brown sugar": "黑糖", "Salt": "鹽巴", "Yeast": "酵母粉", "Baking Powder": "泡打粉", "Vanilla Extract": "香草精", "Chocolate": "巧克力", "Cream Cheese": "奶油乳酪", "Matcha Powder": "抹茶粉",
    "Sesame Oil": "香油", "Oyster Sauce": "蠔油", "Rice Wine": "米酒", "Cooking wine": "料理酒", "Star Anise": "八角", "Cinnamon": "肉桂", "Curry powder": "咖哩粉", "Pepper powder": "胡椒粉", "Black pepper": "黑胡椒",
    "Dried Shrimp": "蝦米", "Fermented Black Beans": "豆豉", "Doubanjiang": "豆瓣醬", "Miso": "味噌", "Shacha sauce": "沙茶醬",
    
    # 罐頭與飲料
    "Tuna can": "鮪魚罐頭", "Kimchi can": "泡菜罐頭", "Corn can": "玉米罐頭", "Tomato can": "番茄罐頭", "Pickled cucumber": "脆瓜罐頭",
    "Coke": "可樂", "Pepsi": "百事可樂", "Sprite": "雪碧", "Milk tea": "奶茶", "Oolong tea": "烏龍茶", "Green tea": "綠茶", "Black tea": "紅茶", "Coffee": "咖啡", "Juice": "果汁", "Beer": "啤酒", "Wine": "紅酒", "Mineral water": "礦泉水"
}

# 修改後的寫法：改在推論時才過濾類別
model = YOLO('yolov8n.pt')
# 把 model.set_classes(custom_food_list) 這一行刪掉或註解掉

@app.route('/scan_image', methods=['POST'])
def scan_image():
    try:
        # 1. 接收前端傳來的 base64 圖片資料
        data = request.json.get('image')
        if not data:
            return jsonify({"error": "No image data"}), 400

        # 2. 解析 base64 並轉換為 OpenCV 可讀取的格式
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. 進行 YOLO 辨識
        # 🔥 核彈級降維打擊：門檻降到 0.15
        results = model(img, conf=0.15, iou=0.3)
        inventory_count = {}

        # 4. 統計辨識結果
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                if cls_idx < len(custom_food_list):
                    eng_name = custom_food_list[cls_idx]
                    # 透過字典將複雜的英文描述對應回標準中文
                    zh_name = translation_dict.get(eng_name, eng_name)
                    inventory_count[zh_name] = inventory_count.get(zh_name, 0) + 1
                    
        # 5. 回傳 JSON 清單給前端
        return jsonify(inventory_count)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 啟動伺服器
    app.run(host='0.0.0.0', port=5000, debug=False)