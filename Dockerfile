# 一般 Node 伺服器版的 detect（functions/server.js），設定見 server.js 開頭；建置：docker build -t holdscan-detect .
FROM node:22-slim
WORKDIR /app
COPY functions/package.json functions/package-lock.json functions/.npmrc ./
RUN npm ci --omit=dev
COPY functions/*.js ./
COPY core/detect-core.js models/freeclimbs.onnx ./vendor/
ENV PORT=8080
EXPOSE 8080
CMD ["node", "server.js"]
