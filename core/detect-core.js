// 岩點偵測的共用核心：雲端函式（functions/）和瀏覽器版（web/detector.js）都用這一份，兩邊的結果才會一樣。
// 模型：freeclimbs YOLOv8n（models/freeclimbs-1280.onnx，AGPL-3.0），輸入 [1, 3, 1280, 1280]，輸出 [1, 5, N] = cx, cy, w, h, score。
//
// 輸入 1280：用 App 修正過的四面牆評估（holdlab tools/eval_detect_gt.py），比 960 召回 86.8% → 92.4%、精確 78.6% → 81.2%；
// 1600、2048 召回差不多、誤判變多；推論時間約 2.4 倍（2 執行緒 CPU 約 0.4 秒）
//
// 分數門檻 0.25：用岩館實際照片看過，提高會漏掉跟牆面顏色接近的岩點，降低多出來的大多是牆上的字和岩點的影子
// 重複框：重疊比例（IoU）超過 0.5，或一個框有 75% 落在另一個大小相近（面積比 < 2.5）的框裡，就當作同一顆岩點
//（模型常把「岩點」和「岩點＋下方影子」各框一次）；大岩塊上的小岩點面積差很多，不受影響
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.HoldScanCore = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const MODEL = 'freeclimbs-1280';
  const MODEL_SIZE = 1280;
  const CONF = 0.25, IOU = 0.5, INSIDE = 0.75, SAME_SIZE = 2.5;

  // 照片 w×h 要縮成多少（等比縮放貼到左上，其餘補灰 114）
  const scaleFor = (w, h) => MODEL_SIZE / Math.max(w, h);

  // 已經縮好、貼在左上的 RGBA（S×S，補邊是灰 114）→ NCHW float32（0–1）
  function toTensorData(rgba) {
    const n = MODEL_SIZE * MODEL_SIZE, out = new Float32Array(3 * n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      out[i] = rgba[j] / 255; out[n + i] = rgba[j + 1] / 255; out[2 * n + i] = rgba[j + 2] / 255;
    }
    return out;
  }

  // 輸出 [1, 5, N] → 外框（照片座標）+ 去重複框；r＝scaleFor 的縮放比例
  function postprocess(data, r) {
    const N = data.length / 5, cand = [];
    for (let i = 0; i < N; i++) {
      const sc = data[4 * N + i];
      if (!(sc >= CONF)) continue;
      const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
      cand.push([(cx - w / 2) / r, (cy - h / 2) / r, (cx + w / 2) / r, (cy + h / 2) / r, sc]);
    }
    cand.sort((a, b) => b[4] - a[4]);
    const keep = [], dead = new Uint8Array(cand.length);
    for (let i = 0; i < cand.length; i++) {
      if (dead[i]) continue;
      const a = cand[i]; keep.push(a);
      const aa = (a[2] - a[0]) * (a[3] - a[1]);
      for (let j = i + 1; j < cand.length; j++) {
        if (dead[j]) continue;
        const b = cand[j];
        const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
        if (iw <= 0 || ih <= 0) continue;
        const inter = iw * ih, ab = (b[2] - b[0]) * (b[3] - b[1]);
        const small = Math.min(aa, ab), big = Math.max(aa, ab);
        if (inter / (aa + ab - inter) > IOU || (inter / small > INSIDE && big / small < SAME_SIZE)) dead[j] = 1;
      }
    }
    return keep;
  }

  return { MODEL, MODEL_SIZE, CONF, IOU, INSIDE, SAME_SIZE, scaleFor, toTensorData, postprocess };
});
