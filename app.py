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

# --- 1. 官方模型絕對不能改的 80 個順序 (確保對位準確) ---
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

# --- 2. 專題專用翻譯字典 (我們只翻譯食物，其他的當作沒看到) ---
translation_dict = {
    "apple": "蘋果",
    "orange": "橘子",  # 👈 你的橘子在這裡！
    "banana": "香蕉",
    "broccoli": "青花菜",
    "carrot": "紅蘿蔔",
    "sandwich": "三明治",
    "pizza": "披薩",
    "cake": "蛋糕",
    "donut": "甜甜圈",
    "hot dog": "熱狗",
    "bottle": "瓶裝飲料",
    "cup": "杯裝飲品",
    "bowl": "碗裝食物"
}

# 修改後的寫法：改在推論時才過濾類別
model = YOLO('yolov8n.pt')
torch.set_num_threads(1)
# 把 model.set_classes(custom_food_list) 這一行刪掉或註解掉

@app.route('/scan_image', methods=['POST'])
def scan_image():
    try:
        # 1. 確保有收到正確的 JSON 與圖片資料
        json_data = request.get_json()
        if not json_data or 'image' not in json_data:
            print("🚨 錯誤：缺少 image 欄位或非 JSON 格式")
            return jsonify({"error": "Missing image data"}), 400

        data = json_data['image']
        if ',' not in data:
            print("🚨 錯誤：圖片 Base64 格式不正確")
            return jsonify({"error": "Invalid format"}), 400

        # 2. 解析 base64 並轉換為 OpenCV 圖片
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. 進行 YOLO 辨識
        results = model(img, conf=0.15, iou=0.3)
        inventory_count = {}

        # 4. 統計辨識結果 (過濾系統)
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                if cls_idx < len(custom_food_list):
                    eng_name = custom_food_list[cls_idx]
                    
                    # 只有在翻譯字典裡的「食物」才會被加入清單
                    if eng_name in translation_dict:
                        zh_name = translation_dict[eng_name]
                        inventory_count[zh_name] = inventory_count.get(zh_name, 0) + 1
                    
        return jsonify(inventory_count)

    except Exception as e:
        print("🚨 發生嚴重錯誤:", str(e))
        print(traceback.format_exc()) 
        return jsonify({"error": "伺服器內部錯誤", "details": str(e)}), 500

if __name__ == '__main__':
    # 啟動伺服器
    app.run(host='0.0.0.0', port=5000, debug=False)