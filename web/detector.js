// 岩點偵測（瀏覽器端）：onnxruntime-web 跑 freeclimbs YOLOv8n，輸出岩點外框。
// 前後處理在 core/detect-core.js（跟雲端函式共用），要先載入。
//
// 執行方式自我檢查：各家手機 GPU 的 WebGPU 實作有 bug（例如 Adreno 7xx 用 NHWC 會算錯），
// 所以第一次用時拿一張固定的測試圖，先用 WASM 算出參考答案，再依序試 WebGPU NHWC → NCHW，
// 結果一致才採用，都不對就用 WASM。結論依 GPU 型號與 onnxruntime 版本存在 localStorage。
//
// 用法：
//   const r = await HoldDetector.detect(canvas, (msg) => console.log(msg));
//   r.boxes → [[x0, y0, x1, y1, score], ...]（輸入影像座標），r.backend → 'webgpu-NHWC' | 'webgpu-NCHW' | 'wasm'
(() => {
  'use strict';
  const ORT_VERSION = '1.30.0';
  const Core = self.HoldScanCore, MODEL_SIZE = Core.MODEL_SIZE;
  const MODEL_URL = new URL(`../models/${Core.MODEL}.onnx`, document.currentScript.src).href;
  const CHOICE_KEY = 'spraywall.backend.v1';
  const T = (k, ...a) => (self.spraywallT ? self.spraywallT(k, ...a) : k);   // 訊息在語言包（i18n/<語言>.js），用 index.html 的 t()

  let ortReady = null, modelBuf = null, session = null, backend = null;

  function loadOrt() {
    if (!ortReady) {
      ortReady = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.js`;
        s.onload = () => {
          ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
          ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
          resolve();
        };
        s.onerror = () => { ortReady = null; s.remove(); reject(new Error(T('ortLoadFail'))); };
        document.head.append(s);
      });
    }
    return ortReady;
  }

  async function loadModel(status) {
    if (!modelBuf) {
      status(T('detDownload'));
      const res = await fetch(MODEL_URL);
      if (!res.ok) throw new Error(T('modelHttp', res.status));
      modelBuf = await res.arrayBuffer();
    }
    return modelBuf;
  }

  function create(kind) {
    const provider = kind === 'wasm' ? 'wasm' : { name: 'webgpu', preferredLayout: kind.split('-')[1] };
    return ort.InferenceSession.create(modelBuf, { executionProviders: [provider], graphOptimizationLevel: 'all' });
  }

  // 等比縮放貼到左上、補灰邊 114，轉 NCHW float32
  function preprocess(src) {
    const S = MODEL_SIZE, r = Core.scaleFor(src.width, src.height);
    const c = document.createElement('canvas'); c.width = S; c.height = S;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.fillStyle = 'rgb(114,114,114)'; g.fillRect(0, 0, S, S);
    g.drawImage(src, 0, 0, Math.round(src.width * r), Math.round(src.height * r));
    return { tensor: new ort.Tensor('float32', Core.toTensorData(g.getImageData(0, 0, S, S).data), [1, 3, S, S]), r };
  }

  async function run(sess, tensor) {
    const out = await sess.run({ [sess.inputNames[0]]: tensor });
    return out[sess.outputNames[0]].data;
  }

  const postprocess = Core.postprocess;

  // 自我檢查用的固定測試圖（與 web/bench.html 的測試圖相同）
  function canaryImage() {
    const c = document.createElement('canvas'); c.width = 1600; c.height = 1200;
    const g = c.getContext('2d');
    g.fillStyle = '#cfcac0'; g.fillRect(0, 0, c.width, c.height);
    let s = 11; const rnd = () => (s = (s * 9301 + 49297) % 233280) / 233280;
    const cols = ['#d6283b', '#f5a524', '#1f9d55', '#2563eb', '#8b3fd9', '#f2d21b', '#1c1c1c', '#14a3c7', '#f06292'];
    for (let i = 0; i < 260; i++) {
      g.fillStyle = cols[i % cols.length];
      g.beginPath();
      g.ellipse(40 + rnd() * 1520, 40 + rnd() * 1120, 8 + rnd() * 45, 6 + rnd() * 30, rnd() * 3, 0, 7);
      g.fill();
    }
    return c;
  }

  async function gpuKey() {
    if (!navigator.gpu || !self.isSecureContext) return null;
    try {
      const a = await navigator.gpu.requestAdapter();
      if (!a) return null;
      const i = a.info || {};
      return [ORT_VERSION, navigator.userAgent, i.vendor, i.architecture, i.device, i.description].join('|');   // 瀏覽器更新後重新檢查
    } catch (e) { return null; }
  }

  function readChoice(key) {
    try { const c = JSON.parse(localStorage.getItem(CHOICE_KEY) || '{}'); return c[key] || null; } catch (e) { return null; }
  }
  function writeChoice(key, kind) {
    try {
      const c = JSON.parse(localStorage.getItem(CHOICE_KEY) || '{}'); c[key] = kind;
      localStorage.setItem(CHOICE_KEY, JSON.stringify(c));
    } catch (e) { /* 私密模式等，下次重新檢查 */ }
  }

  // 分數通道與參考答案比對
  function agrees(ref, got) {
    const N = ref.length / 5;
    let maxRef = 0, maxDiff = 0;
    for (let i = 4 * N; i < 5 * N; i++) {
      if (!Number.isFinite(got[i])) return false;
      maxRef = Math.max(maxRef, ref[i]);
      maxDiff = Math.max(maxDiff, Math.abs(ref[i] - got[i]));
    }
    return maxRef > 0.5 && maxDiff < 0.05;
  }

  async function pickBackend(status) {
    const key = await gpuKey();
    if (!key) return { kind: 'wasm', sess: await create('wasm') };
    const saved = readChoice(key);
    if (saved) {
      try { return { kind: saved, sess: await create(saved) }; } catch (e) { /* 記住的方式失效，重新檢查 */ }
    }

    status(T('detFirstRun'));
    const { tensor } = preprocess(canaryImage());
    const wasm = await create('wasm');
    const ref = (await run(wasm, tensor)).slice();
    for (const kind of ['webgpu-NHWC', 'webgpu-NCHW']) {
      let sess = null;
      try {
        sess = await create(kind);
        if (agrees(ref, await run(sess, tensor))) {
          await wasm.release();
          writeChoice(key, kind);
          return { kind, sess };
        }
      } catch (e) { /* 這種設定不支援，試下一個 */ }
      if (sess) await sess.release();
    }
    writeChoice(key, 'wasm');
    return { kind: 'wasm', sess: wasm };
  }

  async function ensureSession(status) {
    if (session) return;
    await loadOrt();
    await loadModel(status);
    status(T('detPreparing'));
    const p = await pickBackend(status);
    session = p.sess; backend = p.kind;
  }

  async function detect(src, status = () => {}) {
    await ensureSession(status);
    status(T('detFinding'));
    const t = performance.now();
    const { tensor, r } = preprocess(src);
    const boxes = postprocess(await run(session, tensor), r);
    return { boxes, backend, ms: Math.round(performance.now() - t) };
  }

  // 清掉記住的執行方式（換瀏覽器版本後偵測怪怪的時候用）
  function resetChoice() {
    try { localStorage.removeItem(CHOICE_KEY); } catch (e) { /* 忽略 */ }
    if (session) session.release();
    session = null; backend = null;
  }

  window.HoldDetector = { detect, resetChoice, get backend() { return backend; }, ORT_VERSION, MODEL_SIZE };
})();
