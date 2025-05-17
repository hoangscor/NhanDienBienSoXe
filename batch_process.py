import os
import cv2
import time
import imutils
import pytesseract

OUTPUT_DIR = os.path.join('static', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pipeline_anpr(img_bgr):
    # (copy nguyên hàm này từ code Flask của bạn)
    img = imutils.resize(img_bgr, width=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    bf = cv2.bilateralFilter(eq, 11, 17, 17)
    edged = cv2.Canny(bf, 170, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            screenCnt = approx
            break
    if screenCnt is None:
        return None
    x,y,w,h = cv2.boundingRect(screenCnt)
    plate = gray[y:y+h, x:x+w]
    den = cv2.fastNlMeansDenoising(plate, None, 30,7,21)
    th = cv2.threshold(den,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    inv = cv2.bitwise_not(th)
    ch_cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in ch_cnts:
        x2,y2,w2,h2 = cv2.boundingRect(c)
        if w2*h2>500:
            boxes.append((x2,y2,w2,h2))
    boxes = sorted(boxes, key=lambda b:(b[1]//10, b[0]))
    plate_text = ""
    cfg = "--psm 8 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    for (x2,y2,w2,h2) in boxes:
        roi = inv[y2:y2+h2, x2:x2+w2]
        txt = pytesseract.image_to_string(roi, config=cfg).strip()
        plate_text += txt
    return plate_text

def batch_anpr(test_dir='test', result_file='result.txt', delay=1):
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    with open(result_file, 'w', encoding='utf-8') as rf:
        for idx, fname in enumerate(files):
            img_path = os.path.join(test_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Không đọc được ảnh: {fname}")
                rf.write(f"{fname} biển số : Không đọc được ảnh\n")
                continue
            plate = pipeline_anpr(img)
            if not plate or plate.strip() == '':
                plate = 'Không nhận diện được'
            rf.write(f"{fname} biển số : {plate}\n")
            print(f"[{idx+1}/{len(files)}] {fname} biển số: {plate}")
            time.sleep(delay)
    print("Đã xong. Kết quả lưu ở", result_file)

if __name__ == '__main__':
    batch_anpr(test_dir='test', result_file='result.txt', delay=1)

