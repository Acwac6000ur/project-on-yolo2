# -*- coding: cp1251 -*-
import cv2, os, re, time, pytesseract
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# Настройки путей и директорий
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
out_dir = r'D:\VSpython\project_with_vb2\project_with_vb\PLATES'
os.makedirs(out_dir, exist_ok=True)

# Инициализация моделей
model_detect = YOLO('yolov8n.pt')
model_car = YOLO(r'D:\split\split\runs\classify\train\weights\best.pt')
model_plate = YOLO(r'D:\VSpython\project_with_vb2\license_plate_detector.pt')

# Таблицы замен для исправления OCR ошибок
L2D = {'O':'0','B':'8','A':'4','E':'5','S':'5','I':'1','T':'7','Z':'2','G':'6','H':'4'}
D2L = {'0':'O','8':'B','4':'A','5':'E','2':'Z','7':'T','6':'G','1':'I'}
ENG_TO_RUS = {'A':'А','B':'В','E':'Е','K':'К','M':'М','H':'Н','O':'О','P':'Р','C':'С','T':'Т','Y':'У','X':'Х'}

GOST_MAIN, GOST_REG = '0123456789ABCEHKMOPTYX', '0123456789'
car_brand_history = defaultdict(lambda: defaultdict(float))

def parse_and_fix_gost(main_text, reg_text):
    m_clean = "".join(c for c in main_text.replace(" ", "").upper() if c in GOST_MAIN)
    r_clean = "".join(c if c.isdigit() else L2D.get(c, '0') for c in reg_text.replace(" ", "").upper() if c.isdigit() or c in L2D)
    
    if len(r_clean) < 2 or len(m_clean) < 6: return None
    
    mask = "".join("D" if (c.isdigit() or c in L2D) else "L" for c in m_clean)
    match = re.search(r"D{3}", mask)
    if not match: return None
    
    idx = match.start()
    if idx == 0 or idx + 5 > len(m_clean): return None
    
    reg_number = "".join(m_clean[i] if m_clean[i].isdigit() else L2D.get(m_clean[i], '0') for i in range(idx, idx+3))
    fix_l = lambda c: ENG_TO_RUS.get(c if c.isalpha() else D2L.get(c, 'O'), 'А')
    
    return reg_number, f"{fix_l(m_clean[idx-1])}{fix_l(m_clean[idx+3])}{fix_l(m_clean[idx+4])}", r_clean[:3]

def preprocess_zone(img, scale=3.0):
    if img.size == 0: return img
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    _, thresh = cv2.threshold(cv2.GaussianBlur(clahe, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

cap = cv2.VideoCapture(r"D:\VSpython\project_with_vb2\q.mp4")
last_saved_plate, last_save_time = "", 0
print("YOLOv8 Cascade Car Classifier + Advanced Plate OCR Started...")

while True:
    success, img = cap.read()
    if not success:
        print("Video ended. Rewinding to start...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        car_brand_history.clear()
        continue

    img_h, img_w = img.shape[:2]
    results_car = model_detect.track(img, conf=0.45, classes=[2, 5, 7], persist=True, verbose=False)
    
    for result in results_car:
        if not result.boxes: continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            car_id = int(box.id.item()) if box.id is not None else 0
            
            # Кроп машины с отступами
            bw, bh = x2 - x1, y2 - y1
            cx1, cy1 = max(0, x1 + int(bw * 0.05)), max(0, y1 + int(bh * 0.05))
            cx2, cy2 = min(img_w, x2 - int(bw * 0.05)), min(img_h, y2 - int(bh * 0.02))
            
            crop_car = img[cy1:cy2, cx1:cx2]
            if crop_car.size == 0: continue
            
            # Классификация марки авто
            cls_res = model_car(crop_car, verbose=False)
            top1_idx = cls_res[0].probs.top1
            top1_conf = float(cls_res[0].probs.top1conf.item())
            raw_label = cls_res[0].names[top1_idx]
            
            if top1_conf > 0.50:
                car_brand_history[car_id][raw_label] += top1_conf
            
            if car_brand_history[car_id]:
                best_label = max(car_brand_history[car_id], key=car_brand_history[car_id].get)
                car_label = best_label if car_brand_history[car_id][best_label] > 1.2 else "Unknown"
            else:
                car_label = "Unknown"
            
            # Локализация зоны номера
            roi_y1 = max(0, y1 + int(bh * 0.45))
            roi_car_bottom = img[roi_y1:y2, x1:x2]
            if roi_car_bottom.size == 0: continue
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID {car_id}: {car_label}", (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Детекция номера
            results_plate = model_plate(roi_car_bottom, conf=0.25, verbose=False)
            for r_plate in results_plate:
                for p_box in r_plate.boxes:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0].tolist())
                    if (px2 - px1) < 30 or (py2 - py1) < 10: continue
                    
                    roi_plate = roi_car_bottom[py1:py2, px1:px2]
                    if roi_plate.size == 0: continue
                    
                    cv2.rectangle(img, (x1 + px1, roi_y1 + py1), (x1 + px2, roi_y1 + py2), (0, 0, 255), 2)
                    
                    # Зонирование для OCR
                    h, w = roi_plate.shape[:2]
                    split_x, split_y = int(w * 0.73), int(h * 0.72)
                    crop_main, crop_reg = roi_plate[0:h, 0:split_x], roi_plate[0:split_y, split_x:w]
                    if crop_main.size == 0 or crop_reg.size == 0: continue
                    
                    # Препроцессинг и распознавание
                    t_main = preprocess_zone(crop_main, scale=3.5)
                    t_reg = preprocess_zone(crop_reg, scale=4.5)
                    
                    raw_main = pytesseract.image_to_string(t_main, config=f'--psm 7 -c tessedit_char_whitelist={GOST_MAIN}').strip()
                    raw_reg = pytesseract.image_to_string(t_reg, config=f'--psm 8 -c tessedit_char_whitelist={GOST_REG}').strip()
                    if not raw_main or not raw_reg: continue
                    
                    parsed_data = parse_and_fix_gost(raw_main, raw_reg)
                    if parsed_data is None: continue
                    
                    reg_number, num_series, region_code = parsed_data
                    formatted_plate = f"{num_series[0]}{reg_number}{num_series[1:]} {region_code}"
                    curr_time = time.time()
                    
                    print(f"-> Verified Match. [{car_label}] Main: '{raw_main}' | Reg: '{raw_reg}' | Fixed: {formatted_plate}")
                    
                    # Логика сохранения данных
                    if formatted_plate != last_saved_plate or (curr_time - last_save_time > 3.0):
                        now = datetime.now()
                        uid = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}__id_{int((curr_time % 1) * 1000)}"
                        
                        try:
                            is_success, im_buf = cv2.imencode('.jpg', roi_plate)
                            if is_success:
                                im_buf.tofile(os.path.join(out_dir, f"{uid}.jpg"))
                                with open(os.path.join(out_dir, f"{uid}.txt"), "w", encoding="utf-8") as f:
                                    f.write(f"Код номера: {formatted_plate}\nМарка/Тип авто: {car_label}\nДата создания: {now.strftime('%d.%m.%Y %H:%M:%S')}\n")
                                print(f"!!! VERIFIED DATA SAVED: {uid}.jpg")
                                last_saved_plate, last_save_time = formatted_plate, curr_time
                        except Exception as e:
                            print(f"Save error: {e}")

    cv2.imshow("YOLOv8 Video Stream", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()













