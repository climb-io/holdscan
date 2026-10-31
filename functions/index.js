// Firebase 版的 detect：要登入（Firebase 的 ID token，匿名也可以）才能用；處理本身在 handler.js。
// 每個帳號每天最多偵測 DETECT_LIMIT 次（台北時間零點重算），記在 Firestore quota/{uid} 的 detect: { day, n }；超過回 429 { error: 'quota' }
const { onRequest } = require('firebase-functions/v2/https');
const admin = require('firebase-admin');
const { getFirestore } = require('firebase-admin/firestore');
const { createHandler, detect, CPUS } = require('./handler');

admin.initializeApp();

const ORIGINS = ['https://climbio.web.app', 'https://climbio-dev.web.app', 'https://climbio.firebaseapp.com', 'https://climb-io.github.io', 'http://localhost:8765', 'http://127.0.0.1:8765'];
const PREVIEW = /^https:\/\/climbio--[a-z0-9-]+\.web\.app$/;   // Hosting 預覽網址（測試用）
const DETECT_LIMIT = 30;

// 這個帳號今天還能偵測就記一次、回傳 true
const taipeiDay = (now) => Math.floor((now + 8 * 3600e3) / 864e5);
async function takeQuota(db, uid, now = Date.now()) {
  const ref = db.collection('quota').doc(uid), day = taipeiDay(now);
  return db.runTransaction(async (tx) => {
    const q = await tx.get(ref), d = q.exists && q.get('detect'), used = d && d.day === day ? d.n || 0 : 0;
    if (used >= DETECT_LIMIT) return false;
    tx.set(ref, { detect: { day, n: used + 1 } }, { merge: true });
    return true;
  });
}

exports.detect = onRequest({ region: 'asia-east1', invoker: 'public', memory: '2GiB', cpu: CPUS, timeoutSeconds: 60, concurrency: 1, maxInstances: 5 }, createHandler({
  verify: async (req) => (await admin.auth().verifyIdToken((req.get('authorization') || '').replace(/^Bearer /, ''))).uid,
  takeQuota: (uid) => takeQuota(getFirestore('climbio'), uid),
  allowOrigin: (origin) => ORIGINS.includes(origin) || PREVIEW.test(origin),
}));

exports._detect = detect;   // 本機測試用
exports._takeQuota = takeQuota;
