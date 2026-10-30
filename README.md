# holdscan

岩點偵測：照片 → 岩點框。使用 [jwlarocque/yolov8n-freeclimbs-detect-2](https://huggingface.co/jwlarocque/yolov8n-freeclimbs-detect-2)（freeclimbs YOLOv8n），
因此本專案以 **AGPL-3.0** 授權（見 [LICENSE](LICENSE)、[models/README.md](models/README.md)）。

只負責「照片 → 框」；外框描繪、MobileSAM 重描、顏色計算在 climb-io/holdkit（Apache-2.0），App 在 Climbio。
三者之間只交換資料（框的陣列、岩點資料包），不互相引用程式。

## 結構

| 路徑 | 內容 |
|---|---|
| `core/detect-core.js` | 前處理、後處理（分數門檻、去重複框） |
| `functions/` | 雲端偵測函式 `detect`（Firebase，asia-east1，CPU） |
| `models/freeclimbs.onnx` | 偵測模型（AGPL-3.0） |

## 雲端 API

`POST https://asia-east1-firebase-climbio.cloudfunctions.net/detect`

- 標頭：`Authorization: Bearer <Firebase ID token>`（匿名登入也可以）
- 內容：`{ "img": "<JPEG 或 PNG 的 base64>", "size": 1280 }`（傳原尺寸即可，長邊建議最多 2048；伺服器會縮到模型輸入的 `size`：1280（預設）、2048、2560，越大小岩點抓得越多、越慢。在用戶端先縮小再壓 JPEG 會漏掉幾顆分數接近門檻的岩點）
- 回傳：`{ "model": "freeclimbs-1280"（模型－輸入大小）, "boxes": [[x0, y0, x1, y1, score], ...], "w", "h", "ms", "source" }`，框是收到的照片座標

`GET` 同一個網址＝預熱：執行個體開機、載入模型後回 204（不帶照片、不用登入、不算偵測次數）。

照片只在處理當下存在伺服器記憶體中，不存檔、不留紀錄、不用來訓練模型。

## 授權

AGPL-3.0。透過網路使用本服務的人可以在這裡取得完整原始碼：https://github.com/climb-io/holdscan
