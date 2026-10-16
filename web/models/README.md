# 岩點偵測模型（ONNX）

`freeclimbs-960.onnx` 由 [jwlarocque/yolov8n-freeclimbs-detect-2](https://huggingface.co/jwlarocque/yolov8n-freeclimbs-detect-2)
的 `.pt` 權重以 Ultralytics 匯出（opset 17、固定輸入尺寸），未重新訓練。原權重授權為 AGPL-3.0，本目錄沿用同一授權。

重新匯出：

```bash
.venv/bin/python -c "from ultralytics import YOLO; YOLO('weights/jwlarocque/yolov8n-freeclimbs-detect-2.pt').export(format='onnx', imgsz=960, opset=17)"
```

輸入 `images` [1, 3, S, S] float32（RGB、0–1、左上對齊補灰 114）；輸出 `output0` [1, 5, N] = cx, cy, w, h, score。

## MobileSAM（電腦版的「重描輪廓」，見 `samrefine.js`）

| 檔案 | 內容 |
|---|---|
| `mobilesam-encoder.onnx` | 影像 encoder（輸入 H×W×3、0–255，長邊 1024；正規化和補邊在模型裡），輸出 [1, 256, 64, 64] |
| `mobilesam-decoder.onnx` | SAM mask decoder（單一遮罩）；提示用外框（label 2、3） |

- 權重：[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)（Chaoning Zhang 等），Apache-2.0，授權全文見 `LICENSE-MobileSAM`。
- ONNX：取自 [Acly/MobileSAM](https://huggingface.co/Acly/MobileSAM)（MIT），encoder 用該 repo 的 `export_image_encoder.py`、decoder 用 MobileSAM 的 `export_models.py` 匯出；本專案未修改模型，只改了檔名（`mobile_sam_image_encoder.onnx`、`sam_mask_decoder_single.onnx`）。
- 只有電腦版會下載（第一次按「重描」時，約 45 MB）；手機的 GPU 實測會算錯、CPU 太慢，不開這個功能。
