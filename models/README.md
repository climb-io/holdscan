# 岩點偵測模型（ONNX）

`freeclimbs.onnx` 由 [jwlarocque/yolov8n-freeclimbs-detect-2](https://huggingface.co/jwlarocque/yolov8n-freeclimbs-detect-2)
的 `.pt` 權重以 Ultralytics 匯出（opset 17、輸入尺寸可變），未重新訓練。原權重授權為 AGPL-3.0，本專案（holdscan）沿用同一授權。

重新匯出（在 holdlab 執行，權重下載方式見 holdlab 的 README）：

```bash
.venv/bin/python -c "from ultralytics import YOLO; YOLO('weights/jwlarocque/yolov8n-freeclimbs-detect-2.pt').export(format='onnx', imgsz=1280, opset=17, dynamic=True)"
```

輸入 `images` [1, 3, S, S] float32（RGB、0–1、左上對齊補灰 114）；輸出 `output0` [1, 5, N] = cx, cy, w, h, score。
