// holdscan 雲端偵測：照片 → 岩點框（freeclimbs YOLOv8n，CPU）。
// 要登入（Firebase 的 ID token，匿名也可以）才能用；照片只在記憶體裡算，不存檔。
// 前後處理在 core/detect-core.js（部署時複製進來，見 firebase.json 的 predeploy）
const { onRequest } = require('firebase-functions/v2/https');
const admin = require('firebase-admin');
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const path = require('path');
const Core = require('./vendor/detect-core.js');

admin.initializeApp();

const ORIGINS = ['https://climbio.web.app', 'https://climbio-dev.web.app', 'https://climbio.firebaseapp.com', 'https://climb-io.github.io', 'http://localhost:8765', 'http://127.0.0.1:8765'];
const MAX_BYTES = 6e6, CPUS = 2;
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

exports.detect = onRequest({ region: 'asia-east1', invoker: 'public', memory: '2GiB', cpu: CPUS, timeoutSeconds: 60, concurrency: 1, maxInstances: 5 }, async (req, res) => {
  const origin = req.get('origin');
  if (ORIGINS.includes(origin) || /^https:\/\/climbio--[a-z0-9-]+\.web\.app$/.test(origin || '')) { res.set('Access-Control-Allow-Origin', origin); res.set('Vary', 'Origin'); }
  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'POST');
    res.set('Access-Control-Allow-Headers', 'Authorization, Content-Type');
    res.set('Access-Control-Max-Age', '3600');
    return res.status(204).send('');
  }
  if (req.method !== 'POST') return res.status(405).send('POST only');
  try { await admin.auth().verifyIdToken((req.get('authorization') || '').replace(/^Bearer /, '')); }
  catch (e) { return res.status(401).json({ error: 'not signed in' }); }
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
});

exports._detect = detect;   // 本機測試用
