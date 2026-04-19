import traceback
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch  # ✅ 加入這行

app = Flask(__name__)
CORS(app)

# --- 1. 擴充版食材清單 (YOLO 英文標籤) ---
# app.py 修正版：確保前 80 個順序與官方模型同步
custom_food_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

## app.py 中的翻譯字典：將官方 80 種標籤對應為中文
translation_dict = {
    # --- 食材與餐飲相關 (專題展示重點) ---
    "bottle": "瓶裝飲料", "wine glass": "高腳杯", "cup": "杯子", "fork": "叉子", "knife": "刀子", 
    "spoon": "湯匙", "bowl": "碗/容器", "banana": "香蕉", "apple": "蘋果", "sandwich": "三明治", 
    "orange": "橘子", "broccoli": "青花菜", "carrot": "紅蘿蔔", "hot dog": "熱狗", "pizza": "披薩", 
    "donut": "甜甜圈", "cake": "蛋糕", "refrigerator": "冰箱", "microwave": "微波爐", "oven": "烤箱", 
    "toaster": "烤麵包機", "sink": "水槽",

    # --- 其他常見物體 (確保系統穩定性) ---
    "person": "人", "bicycle": "腳踏車", "car": "汽車", "motorcycle": "機車", "airplane": "飛機", 
    "bus": "巴士", "train": "火車", "truck": "卡車", "boat": "船", "traffic light": "紅綠燈", 
    "fire hydrant": "消防栓", "stop sign": "停止標誌", "parking meter": "停車收費桿", "bench": "長椅", 
    "bird": "鳥", "cat": "貓", "dog": "狗", "horse": "馬", "sheep": "羊", "cow": "牛", 
    "elephant": "大象", "bear": "熊", "zebra": "斑馬", "giraffe": "長頸鹿", "backpack": "背包", 
    "umbrella": "雨傘", "handbag": "手提包", "tie": "領帶", "suitcase": "行李箱", "frisbee": "飛盤", 
    "skis": "滑雪板", "snowboard": "滑雪單板", "sports ball": "球類", "kite": "風箏", 
    "baseball bat": "棒球棒", "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "衝浪板", 
    "tennis racket": "網球拍", "chair": "椅子", "couch": "沙發", "potted plant": "盆栽", "bed": "床", 
    "dining table": "餐桌", "toilet": "馬桶", "tv": "電視", "laptop": "筆記型電腦", "mouse": "滑鼠", 
    "remote": "遙控器", "keyboard": "鍵盤", "cell phone": "手機", "book": "書本", "clock": "時鐘", 
    "vase": "花瓶", "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹風機", "toothbrush": "牙刷"
}

# 修改後的寫法：改在推論時才過濾類別
model = YOLO('yolov8n.pt')
torch.set_num_threads(1)
# 把 model.set_classes(custom_food_list) 這一行刪掉或註解掉

@app.route('/scan_image', methods=['POST'])
def scan_image():
    try:
        # 1. 先確認有沒有收到 JSON 資料
        json_data = request.get_json()
        if not json_data:
            print("🚨 錯誤：收到的請求不是 JSON 格式")
            return jsonify({"error": "Missing JSON body"}), 400
            
        # 2. 確認有沒有 image 這個欄位
        data = json_data.get('image')
        if not data:
            print("🚨 錯誤：JSON 中缺少 image 欄位")
            return jsonify({"error": "Missing image field"}), 400

        # 3. 確保資料包含 Base64 的標頭
        if ',' not in data:
            print("🚨 錯誤：圖片格式不正確，缺少逗號")
            return jsonify({"error": "Invalid image format"}), 400
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
        # 這裡會把詳細的錯誤原因（哪一行出錯、什麼錯誤）印在 Render 的 Log 裡
        print("🚨 發生嚴重錯誤:", str(e))
        print(traceback.format_exc()) 
        
        # 回傳給手機前端的錯誤訊息，也多加一個 details 欄位方便查看
        return jsonify({
            "error": "伺服器內部錯誤", 
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # 啟動伺服器
    app.run(host='0.0.0.0', port=5000, debug=False)