# Chess Board Tracking with YOLO11

Hệ thống tracking bàn cờ tích hợp hiệu chuẩn bàn cờ (calibration) và nhận diện quân cờ bằng YOLO11.

## 📋 Cấu trúc Dự án

```
chess-tracking/
├── calibrate_manual_oriented.py    # Hiệu chuẩn bàn cờ thủ công
├── piece_detector.py               # Nhận diện quân cờ với YOLO11
├── train_chess_model.py            # Huấn luyện mô hình YOLO11
├── chess_tracker.py                # Tích hợp hoàn chỉnh (calibration + detection)
├── main.py                         # Phát hiện các cạnh bàn cờ
├── sqdict.json                     # Dữ liệu hiệu chuẩn (tọa độ các ô vuông)
├── requirement.txt                 # Các gói Python cần thiết
└── README.md                       # File này
```

## 🚀 Cài đặt

### 1. Cài đặt các gói phụ thuộc

```bash
pip install -r requirement.txt
```

### 2. GPU Support (Optional)

Để sử dụng GPU cho YOLO11:

```bash
# NVIDIA CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Hướng dẫn Sử dụng

### Bước 1: Hiệu chuẩn Bàn Cờ

Chạy script hiệu chuẩn để tạo file `sqdict.json` chứa tọa độ các ô vuông:

```bash
# Kamera từ phía trước (0°)
python calibrate_manual_oriented.py --rotate 0

# Kamera từ bên phải (90°)
python calibrate_manual_oriented.py --rotate 90

# Kamera từ phía sau (180°)
python calibrate_manual_oriented.py --rotate 180

# Kamera từ bên trái (270°)
python calibrate_manual_oriented.py --rotate 270
```

**Hướng dẫn:**

1. Arahkan kamera ke bàn cờ
2. Klik vào 4 góc của bàn cờ theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái
3. Nhấn `s` để lưu hoặc `r` để reset, `q` để thoát

**Output:** `sqdict.json` (chứa tọa độ 64 ô vuông)

### Bước 2: Chuẩn bị Mô hình YOLO11

#### Option A: Sử dụng Mô hình Có sẵn (Nhanh)

```bash
# Sử dụng mô hình YOLO11 nano (mặc định)
python chess_tracker.py --model yolov8n.pt
```

**Lưu ý:** Mô hình mặc định được huấn luyện trên COCO dataset. Để nhận diện quân cờ cụ thể, cần huấn luyện mô hình riêng.

#### Option B: Huấn luyện Mô hình Riêng (Tốt nhất)

**Chuẩn bị Dataset:**

Tạo cấu trúc thư mục:

```
chess_dataset/
├── images/
│   ├── train/     # ~70% ảnh training
│   ├── val/       # ~15% ảnh validation
│   └── test/      # ~15% ảnh test
├── labels/
│   ├── train/     # YOLO format labels (.txt)
│   ├── val/
│   └── test/
└── data.yaml      # Configuration file
```

**File `data.yaml`:**

```yaml
path: /absolute/path/to/chess_dataset
train: images/train
val: images/val
test: images/test

nc: 12 # Số lớp (12 loại quân cờ)
names:
  [
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
  ]
```

**Huấn luyện:**

```bash
python train_chess_model.py --dataset chess_dataset/data.yaml --epochs 100 --batch-size 16

# Sử dụng GPU (device 0)
python train_chess_model.py --dataset chess_dataset/data.yaml --device 0

# CPU mode
python train_chess_model.py --dataset chess_dataset/data.yaml --device -1
```

**Validate mô hình:**

```bash
python train_chess_model.py --validate chess_models/chess_pieces/weights/best.pt --dataset chess_dataset/data.yaml
```

**Output:** Mô hình được lưu tại `chess_models/chess_pieces/weights/best.pt`

### Bước 3: Tracking Quân Cờ

#### Sử dụng Mô hình Mặc định:

```bash
python chess_tracker.py --sqdict sqdict.json --rotate 0 --confidence 0.5
```

#### Sử dụng Mô hình Huấn Luyện:

```bash
python chess_tracker.py --sqdict sqdict.json --model chess_models/chess_pieces/weights/best.pt --rotate 0
```

#### Các Tùy Chọn:

```bash
python chess_tracker.py \
  --sqdict sqdict.json \
  --model yolov8n.pt \
  --rotate 90 \
  --confidence 0.5 \
  --save session_data.json \
  --no-overlay
```

**Tham số:**

- `--sqdict`: Đường dẫn file hiệu chuẩn JSON
- `--model`: Mô hình YOLO (yolov8n, yolov8s, yolov8m, hoặc đường dẫn file)
- `--rotate`: Xoay kamera (0/90/180/270)
- `--confidence`: Ngưỡng tin cậy (0.0-1.0)
- `--save`: Tên file lưu dữ liệu phiên làm việc
- `--no-overlay`: Tắt hiển thị lưới bàn cờ

**Điều khiển (Khi chạy):**

- `q` - Thoát và lưu phiên làm việc
- `s` - In trạng thái bàn cờ hiện tại
- `c` - Xóa lịch sử bước đi
- `p` - In lịch sử các bước đi

## 🔍 Chi Tiết Từng Module

### `piece_detector.py`

Class `ChessPieceDetector` cung cấp:

- `detect_pieces(frame, confidence_threshold)`: Phát hiện quân cờ trong khung hình
- `map_pieces_to_squares(detections)`: Ánh xạ quân cờ vào các ô vuông
- `detect_moves(current_positions)`: Phát hiện bước đi quân cờ
- `draw_detections(frame, detections, piece_positions)`: Vẽ kết quả lên khung hình
- `process_frame(frame, draw)`: Pipeline hoàn chỉnh

**Ví dụ sử dụng:**

```python
from piece_detector import ChessPieceDetector
import cv2

detector = ChessPieceDetector(sqdict_path='sqdict.json', model_name='yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = detector.process_frame(frame, draw=True)

    # result['positions']: {square_name: {class, confidence, center, bbox}}
    # result['moves']: [{from, to, piece, type}]
    # result['frame']: Annotated frame

    cv2.imshow('Detection', result['frame'])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### `chess_tracker.py`

Class `ChessBoardTracker` tích hợp:

- Hiệu chuẩn bàn cờ (sqdict)
- Nhận diện quân cờ (YOLO11)
- Tracking bước đi
- Lưu lịch sử phiên làm việc

**Ví dụ sử dụng:**

```python
from chess_tracker import ChessBoardTracker
import cv2

tracker = ChessBoardTracker(sqdict_path='sqdict.json', cam_rot=0)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    result = tracker.process_frame(frame)

    # Get board state
    board_state = tracker.get_board_state()  # {square: {class, confidence, ...}}

    # Get move history
    history = tracker.get_move_history()  # [{from, to, piece, type, frame}]

    cv2.imshow('Tracking', result['frame'])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tracker.save_session('my_session.json')
        break

cap.release()
cv2.destroyAllWindows()
```

## 📊 Output Format

### `sqdict.json` (Calibration Data)

```json
{
  "a1": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "a2": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  ...
}
```

### `session_data.json` (Tracking Results)

```json
{
  "total_frames": 1234,
  "total_moves": 5,
  "camera_rotation": 0,
  "calibration_file": "sqdict.json",
  "move_history": [
    {
      "from": "e2",
      "to": "e4",
      "piece": "pawn",
      "type": "move",
      "frame": 45
    },
    {
      "from": "e7",
      "to": "e5",
      "piece": "pawn",
      "type": "move",
      "frame": 120
    }
  ],
  "final_board_state": {
    "e4": {"class": "pawn", "confidence": 0.92, "center": [x, y], "bbox": [x1, y1, x2, y2]},
    "e5": {"class": "pawn", "confidence": 0.88, "center": [x, y], "bbox": [x1, y1, x2, y2]}
  }
}
```

## 🎯 Các Loại Quân Cờ Hỗ Trợ

Mô hình hỗ trợ 12 loại quân cờ (6 loại mỗi màu):

- `white_pawn` / `black_pawn` - Tốt
- `white_knight` / `black_knight` - Mã
- `white_bishop` / `black_bishop` - Tượng
- `white_rook` / `black_rook` - Xe
- `white_queen` / `black_queen` - Hậu
- `white_king` / `black_king` - Vua

## ⚙️ Tùy Chỉnh

### Thay Đổi Confidence Threshold

```bash
# Chỉ chấp nhận detections có confidence >= 0.7
python chess_tracker.py --confidence 0.7
```

### Sử Dụng Mô hình Khác Nhau

```bash
# Nano (nhanh, ít chính xác)
python chess_tracker.py --model yolov8n.pt

# Small (cân bằng)
python chess_tracker.py --model yolov8s.pt

# Medium (chính xác hơn, chậm hơn)
python chess_tracker.py --model yolov8m.pt

# Large (rất chính xác, rất chậm)
python chess_tracker.py --model yolov8l.pt
```

### Tắt Board Overlay

```bash
python chess_tracker.py --no-overlay
```

## 🐛 Troubleshooting

### Lỗi: "Calibration file not found"

```
Giải pháp: Chạy calibrate_manual_oriented.py trước
python calibrate_manual_oriented.py --rotate 0
```

### Lỗi: "CUDA out of memory"

```
Giải pháp: Sử dụng mô hình nhỏ hơn hoặc chạy trên CPU
python chess_tracker.py --model yolov8n.pt
```

### Quân cờ không được phát hiện

```
Giải pháp:
1. Tăng light trong phòng
2. Điều chỉnh confidence threshold thấp hơn
   python chess_tracker.py --confidence 0.3
3. Huấn luyện mô hình với dataset bàn cờ
```

### Bước đi không được detect

```
Giải pháp:
1. Đảm bảo quân cờ được phát hiện (xem board state với 's')
2. Giảm confidence threshold
3. Di chuyển quân cờ chậm hơn để camera kịp theo dõi
```

## 📚 Tài Liệu Tham Khảo

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 License

MIT License

## 🤝 Contribute

Góp ý và báo cáo lỗi tại Issues section
