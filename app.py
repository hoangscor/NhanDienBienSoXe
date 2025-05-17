import os
import threading
import time
import uuid
from flask import Flask, render_template, request, redirect, jsonify
import cv2
import numpy as np
import imutils
import pytesseract
import pymysql
from datetime import datetime

# ===== CẤU HÌNH DATABASE =====
DB_HOST = 'localhost'
DB_PORT = 3308
DB_USER = 'root'
DB_PASS = 'hoangcute123'
DB_NAME = 'parking_system'

# ===== ĐƯỜNG DẪN =====
IMAGE_DIR = 'images'
OUTPUT_DIR = os.path.join('static', 'processed')
CAR_IN_DIR = os.path.join('static', 'car_in')
CAR_OUT_DIR = os.path.join('static', 'car_out')
for d in [OUTPUT_DIR, CAR_IN_DIR, CAR_OUT_DIR, IMAGE_DIR]:
    os.makedirs(d, exist_ok=True)

# ===== KẾT NỐI DATABASE =====
conn = pymysql.connect(
    host=DB_HOST, port=DB_PORT,
    user=DB_USER, password=DB_PASS,
    db=DB_NAME, charset='utf8mb4', autocommit=True
)
cursor = conn.cursor()
# Tạo bảng nếu chưa có, thêm cột nếu thiếu
cursor.execute("""
CREATE TABLE IF NOT EXISTS parking_log (
  id INT AUTO_INCREMENT PRIMARY KEY,
  plate VARCHAR(20),
  entry_time DATETIME,
  exit_time DATETIME,
  fee INT,
  image_in VARCHAR(255),
  image_out VARCHAR(255)
)
""")
def add_column_if_not_exists(table, column, coltype):
    cursor.execute("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
    """, (DB_NAME, table, column))
    exists = cursor.fetchone()[0]
    if exists == 0:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

add_column_if_not_exists("parking_log", "image_in", "VARCHAR(255)")
add_column_if_not_exists("parking_log", "image_out", "VARCHAR(255)")
conn.commit()

# ===== BIẾN TOÀN CỤC TỰ ĐỘNG =====
AUTO_MODE = False
PROCESSED_IMAGES = set()
LAST_AUTO_PLATE = None
LAST_AUTO_ACTION = None
LAST_AUTO_TIME = None

app = Flask(__name__)

def save_entry_image(img, plate, is_in=True):
    folder = CAR_IN_DIR if is_in else CAR_OUT_DIR
    os.makedirs(folder, exist_ok=True)
    fname = f"{plate}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    fpath = os.path.join(folder, fname)
    cv2.imwrite(fpath, img)
    # trả về đường dẫn để lưu vào db (bắt đầu từ static/)
    relpath = os.path.relpath(fpath, start='.')
    return relpath.replace('\\','/')

def pipeline_anpr(img_bgr):
    # 01) Resize → 600px width
    img = imutils.resize(img_bgr, width=600)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'01_resized.jpg'), img)
    # 02) Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'02_gray.jpg'), gray)
    # 03) Equalize histogram
    eq = cv2.equalizeHist(gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'03_equalize.jpg'), eq)
    # 04) Bilateral filter
    bf = cv2.bilateralFilter(eq, 11, 17, 17)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'04_bilateral.jpg'), bf)
    # 05) Canny edge detector
    edged = cv2.Canny(bf, 170, 200)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'05_canny.jpg'), edged)
    # 06) Find top-10 contours by area
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            screenCnt = approx
            break
    # 07) Draw contour
    cont_img = img.copy()
    if screenCnt is not None:
        cv2.drawContours(cont_img, [screenCnt], -1, (0,255,0), 3)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'06_contour.jpg'), cont_img)
    if screenCnt is None:
        return None
    # 08) Crop plate from gray
    x,y,w,h = cv2.boundingRect(screenCnt)
    plate = gray[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(OUTPUT_DIR,'07_crop.jpg'), plate)
    # 09) Denoise + Threshold
    den = cv2.fastNlMeansDenoising(plate, None, 30,7,21)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'08_denoise.jpg'), den)
    th = cv2.threshold(den,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    cv2.imwrite(os.path.join(OUTPUT_DIR,'09_thresh.jpg'), th)
    # 10) Character contours & sort
    inv = cv2.bitwise_not(th)
    ch_cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in ch_cnts:
        x2,y2,w2,h2 = cv2.boundingRect(c)
        if w2*h2>500:
            boxes.append((x2,y2,w2,h2))
    boxes = sorted(boxes, key=lambda b:(b[1]//10, b[0]))
    chars_img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    for (x2,y2,w2,h2) in boxes:
        cv2.rectangle(chars_img,(x2,y2),(x2+w2,y2+h2),(0,255,0),1)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'10_chars.jpg'), chars_img)
    # 11) OCR từng ký tự
    plate_text = ""
    cfg = "--psm 8 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    for (x2,y2,w2,h2) in boxes:
        roi = inv[y2:y2+h2, x2:x2+w2]
        txt = pytesseract.image_to_string(roi, config=cfg).strip()
        plate_text += txt
    # Result overlay
    res = img.copy()
    cv2.drawContours(res, [screenCnt], -1, (0,255,0), 2)
    cv2.putText(res, plate_text, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'11_result.jpg'), res)
    return plate_text

def auto_processing_thread():
    global AUTO_MODE, PROCESSED_IMAGES, LAST_AUTO_PLATE, LAST_AUTO_ACTION, LAST_AUTO_TIME
    while True:
        if AUTO_MODE:
            files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            new_files = [f for f in files if f not in PROCESSED_IMAGES]
            for file in new_files:
                img_path = os.path.join(IMAGE_DIR, file)
                img = cv2.imread(img_path)
                if img is None: continue
                plate = pipeline_anpr(img)
                now = datetime.now()
                if plate:
                    # Kiểm tra xe đã vào mà chưa ra => cho xe ra, lưu ảnh ra
                    cursor.execute(
                        "SELECT id FROM parking_log WHERE plate=%s AND exit_time IS NULL",
                        (plate,)
                    )
                    rec = cursor.fetchone()
                    if rec:
                        img_out_path = save_entry_image(img, plate, False)
                        cursor.execute(
                            "UPDATE parking_log SET exit_time=%s, fee=%s, image_out=%s WHERE id=%s",
                            (now, 5000, img_out_path, rec[0])
                        )
                        LAST_AUTO_ACTION = 'Xe ra'
                    else:
                        img_in_path = save_entry_image(img, plate, True)
                        cursor.execute(
                            "INSERT INTO parking_log(plate,entry_time,image_in) VALUES(%s,%s,%s)",
                            (plate, now, img_in_path)
                        )
                        LAST_AUTO_ACTION = 'Xe vào'
                    conn.commit()
                    LAST_AUTO_PLATE = plate
                    LAST_AUTO_TIME = now.strftime('%H:%M:%S %d-%m-%Y')
                PROCESSED_IMAGES.add(file)
        time.sleep(2)

threading.Thread(target=auto_processing_thread, daemon=True).start()

@app.route('/', methods=['GET', 'POST'])
def index():
    global AUTO_MODE
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    plate, msg = None, None
    if request.method=='POST' and not AUTO_MODE:
        chosen = request.form.get('image_name')
        if chosen and chosen in images:
            img = cv2.imread(os.path.join(IMAGE_DIR, chosen))
            msg = f"Xử lý ảnh mẫu: {chosen}"
        else:
            file = request.files.get('photo')
            if not file:
                msg = "Vui lòng chọn ảnh mẫu hoặc upload/chụp mới!"
                return render_template('index.html', images=images, plate=plate, msg=msg, auto=AUTO_MODE)
            data = file.read()
            img = cv2.imdecode(np.frombuffer(data,np.uint8), cv2.IMREAD_COLOR)
            msg = "Xử lý ảnh upload/chụp"
        plate = pipeline_anpr(img)
        now = datetime.now()
        if not plate:
            msg += " → Không nhận diện được!"
        else:
            msg += f" → Biển số: {plate}"
            cursor.execute(
                "SELECT id FROM parking_log WHERE plate=%s AND exit_time IS NULL",
                (plate,)
            )
            rec = cursor.fetchone()
            if rec:
                img_out_path = save_entry_image(img, plate, False)
                cursor.execute(
                    "UPDATE parking_log SET exit_time=%s, fee=%s, image_out=%s WHERE id=%s",
                    (now, 5000, img_out_path, rec[0])
                )
            else:
                img_in_path = save_entry_image(img, plate, True)
                cursor.execute(
                    "INSERT INTO parking_log(plate,entry_time,image_in) VALUES(%s,%s,%s)",
                    (plate, now, img_in_path)
                )
            conn.commit()
    return render_template('index.html', images=images, plate=plate, msg=msg, auto=AUTO_MODE)

@app.route('/toggle_auto', methods=['POST'])
def toggle_auto():
    global AUTO_MODE
    AUTO_MODE = not AUTO_MODE
    return jsonify({'auto': AUTO_MODE})

@app.route('/get_last_auto')
def get_last_auto():
    return jsonify({
        'plate': LAST_AUTO_PLATE,
        'action': LAST_AUTO_ACTION,
        'time': LAST_AUTO_TIME
    })

@app.route('/upload_auto_image', methods=['POST'])
def upload_auto_image():
    if not AUTO_MODE: return 'ERR', 400
    file = request.data
    fname = f"webcam_{int(time.time())}.jpg"
    fpath = os.path.join(IMAGE_DIR, fname)
    with open(fpath, 'wb') as f:
        f.write(file)
    return 'OK', 200

@app.route('/admin')
def admin():
    # image_in = r[3], image_out = r[5]
    cursor.execute("SELECT id,plate,entry_time,image_in,exit_time,image_out,fee FROM parking_log")
    rows = cursor.fetchall()
    return render_template('admin.html', rows=rows)

@app.route('/delete/<int:rowid>', methods=['POST'])
def delete(rowid):
    cursor.execute("DELETE FROM parking_log WHERE id=%s", (rowid,))
    conn.commit()
    return redirect('/admin')

@app.route('/xera/<int:row_id>')
def xera(row_id):
    now = datetime.now()
    # Lấy lại ảnh vào để làm ảnh ra nếu cần
    cursor.execute("SELECT plate, image_in FROM parking_log WHERE id=%s", (row_id,))
    result = cursor.fetchone()
    plate = result[0]
    img_in = result[1]
    # Cập nhật ảnh ra (có thể dùng lại ảnh vào nếu không có ảnh mới)
    cursor.execute(
        "UPDATE parking_log SET exit_time=%s, fee=%s, image_out=%s WHERE id=%s",
        (now, 5000, img_in, row_id)
    )
    conn.commit()
    return redirect('/admin')

# (Tùy chọn) Sửa thông tin xe
@app.route('/edit/<int:row_id>', methods=['POST'])
def edit(row_id):
    data = request.get_json()
    plate = data.get("plate")
    entry_time = data.get("entry_time")
    exit_time = data.get("exit_time")
    fee = data.get("fee")
    cursor.execute("""
        UPDATE parking_log SET plate=%s, entry_time=%s, exit_time=%s, fee=%s WHERE id=%s
    """, (plate, entry_time, exit_time if exit_time else None, fee, row_id))
    conn.commit()
    return "OK"

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
