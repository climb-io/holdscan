# holdscan

岩點偵測：照片 → 岩點框。使用 [jwlarocque/yolov8n-freeclimbs-detect-2](https://huggingface.co/jwlarocque/yolov8n-freeclimbs-detect-2)（freeclimbs YOLOv8n），
因此本專案以 **AGPL-3.0** 授權（見 [LICENSE](LICENSE)、[models/README.md](models/README.md)）。

只負責「照片 → 框」；外框描繪、MobileSAM 重描、顏色計算在 climb-io/holdkit（Apache-2.0），App 在 Climbio。
三者之間只交換資料（框的陣列、岩點資料包），不互相引用程式。

## 結構

| 路徑 | 內容 |
|---|---|
| `core/detect-core.js` | 前處理、後處理（分數門檻、去重複框） |
| `functions/` | 雲端偵測 `detect`：處理在 `handler.js`（跟平台無關），外殼有 Firebase 版 `index.js`（asia-east1，CPU，每個帳號每天 30 次）和一般 Node 伺服器版 `server.js` |
| `Dockerfile` | `server.js` 的容器映像 |
| `models/freeclimbs.onnx` | 偵測模型（AGPL-3.0） |

## 雲端 API

`POST https://asia-east1-firebase-climbio.cloudfunctions.net/detect`

- 標頭：`Authorization: Bearer <Firebase ID token>`（匿名登入也可以）
- 內容：`{ "img": "<JPEG 或 PNG 的 base64>", "size": 1280 }`（傳原尺寸即可，長邊建議最多 2048；伺服器會縮到模型輸入的 `size`：1280（預設）、2048、2560，越大小岩點抓得越多、越慢。在用戶端先縮小再壓 JPEG 會漏掉幾顆分數接近門檻的岩點）
- 回傳：`{ "model": "freeclimbs-1280"（模型－輸入大小）, "boxes": [[x0, y0, x1, y1, score], ...], "w", "h", "ms", "source" }`，框是收到的照片座標

`GET` 同一個網址＝預熱：執行個體開機、載入模型後回 204（不帶照片、不用登入、不算偵測次數）。

照片只在處理當下存在伺服器記憶體中，不存檔、不留紀錄、不用來訓練模型。

## 自己架設

`functions/server.js` 是同一個服務的一般 Node 伺服器版，不需要 Firebase，Docker、Cloud Run、Fly 或自己的機器都能跑（記憶體至少 2 GB）：

```bash
docker build -t holdscan-detect .
docker run -p 8080:8080 -e API_KEY=換成你的密鑰 -e ORIGINS=https://你的網站 holdscan-detect
```

不用 Docker：

```bash
mkdir -p functions/vendor && cp core/detect-core.js models/freeclimbs.onnx functions/vendor/
cd functions && npm install && API_KEY=換成你的密鑰 node server.js
```

| 環境變數 | 說明 |
|---|---|
| `AUTH` | `key`（預設，`Authorization: Bearer` 要等於 `API_KEY`）、`firebase`（Firebase 的 ID token）或 `none`（不驗證，只在本機或私人網路用） |
| `API_KEY` | `AUTH=key` 的密鑰 |
| `ORIGINS` | 允許跨網域呼叫的網站，逗號分隔；`*` 是全部 |
| `DETECT_CPUS` | 推論執行緒數，照分到的 vCPU 設（預設 2） |
| `PORT` | 預設 8080 |

一般伺服器版沒有每日次數限制；公開架設時用 `AUTH=key`，或在前面的閘道限流。依 AGPL-3.0，改過的版本透過網路提供服務時，要讓使用者取得你的原始碼（回傳的 `source` 改成你的 repo）。

## 授權

AGPL-3.0。透過網路使用本服務的人可以在這裡取得完整原始碼：https://github.com/climb-io/holdscan
