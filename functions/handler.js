// detect 的 HTTP 處理，跟平台無關：Firebase 函式（index.js）和一般 Node 伺服器（server.js）共用。
// 照片 → 岩點框（freeclimbs YOLOv8n，CPU）；照片只在記憶體裡算，不存檔。前後處理在 core/detect-core.js（Firebase 部署時複製進 vendor/，見 firebase.json 的 predeploy；Docker 見 Dockerfile）
// verify(req)：驗證呼叫者、回傳 uid，失敗就丟錯（回 401）；takeQuota(uid)：這次還能不能偵測（false 回 429，沒給就不限次數）；allowOrigin(origin)：這個網站能不能跨網域呼叫
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const path = require('path');
const Core = require('./vendor/detect-core.js');

const MAX_BYTES = 6e6, CPUS = Number(process.env.DETECT_CPUS) || 2;
let session = null;

async function getSession() {
  if (!session) session = await ort.InferenceSession.create(path.join(__dirname, 'vendor', `${Core.MODEL}.onnx`),
    { executionProviders: ['cpu'], graphOptimizationLevel: 'all', intraOpNumThreads: CPUS, interOpNumThreads: 1 });
  return session;
}

// JPEG/PNG → 等比縮放貼到左上、補灰 114 → 偵測，回傳照片座標的框 [x0, y0, x1, y1, score]
async function detect(buf, S = Core.MODEL_SIZE) {
  const meta = await sharp(buf).metadata(), r = Core.scaleFor(meta.width, meta.height, S);
  const rgba = await sharp(buf).removeAlpha().resize(Math.round(meta.width * r), Math.round(meta.height * r), { fit: 'fill' })
    .extend({ right: S - Math.round(meta.width * r), bottom: S - Math.round(meta.height * r), background: { r: 114, g: 114, b: 114 } })
    .ensureAlpha().raw().toBuffer();
  const s = await getSession(), t0 = Date.now();
  const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', Core.toTensorData(rgba, S), [1, 3, S, S]) });
  return { boxes: Core.postprocess(out[s.outputNames[0]].data, r), w: meta.width, h: meta.height, ms: Date.now() - t0 };
}

function createHandler({ verify, takeQuota = async () => true, allowOrigin }) {
  return async (req, res) => {
    const origin = req.get('origin');
    if (origin && allowOrigin(origin)) { res.set('Access-Control-Allow-Origin', origin); res.set('Vary', 'Origin'); }
    if (req.method === 'OPTIONS') {
      res.set('Access-Control-Allow-Methods', 'POST');
      res.set('Access-Control-Allow-Headers', 'Authorization, Content-Type');
      res.set('Access-Control-Max-Age', '3600');
      return res.status(204).send('');
    }
    // 預熱：App 在使用者選照片時送一個 GET，執行個體先開機、載入模型（不帶照片、不用登入、不算偵測次數）
    if (req.method === 'GET') {
      try { await getSession(); return res.status(204).send(''); } catch (e) { console.error(e); return res.status(500).send(''); }
    }
    if (req.method !== 'POST') return res.status(405).send('POST only');
    let uid;
    try { uid = await verify(req); } catch (e) { return res.status(401).json({ error: 'not signed in' }); }
    try { if (!(await takeQuota(uid))) return res.status(429).json({ error: 'quota' }); }
    catch (e) { console.error(e); return res.status(500).json({ error: 'detect failed' }); }
    try {
      const { img, size = Core.MODEL_SIZE } = req.body || {};
      if (typeof img !== 'string' || !Core.SIZES.includes(size)) return res.status(400).json({ error: 'bad request' });
      const buf = Buffer.from(img, 'base64');
      if (buf.length > MAX_BYTES) return res.status(413).json({ error: 'image too large' });
      res.json({ model: `${Core.MODEL}-${size}`, ...(await detect(buf, size)), source: 'https://github.com/climb-io/holdscan' });
    } catch (e) {
      console.error(e);   // 詳細原因只留在伺服器紀錄，不回給用戶端
      res.status(500).json({ error: 'detect failed' });
    }
  };
}

module.exports = { createHandler, detect, CPUS };
