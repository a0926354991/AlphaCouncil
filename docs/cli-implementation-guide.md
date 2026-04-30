# AlphaCouncil 排程落檔實作指引（純規格）

本文件定義 **AlphaCouncil 部署到雲端並定時產生報告寫入 GCS/本地** 的實作規格。
重點是「可運維、可觀測、可重跑」，並採用已確認策略：

- 排程模式：同步單次 Job
- 完成判準：最後一個 agent 結束 callback 寫檔（Sequential workflow）
- 產出粒度：單一整合報告

---

## 1. 目標與範圍

### 1.1 目標

建立一個可由排程系統觸發的 orchestrator，流程如下：

1. 接收 ticker/market 等參數
2. 呼叫 AlphaCouncil Run API 啟動單次 workflow run
3. 由最後一個 agent callback 觸發報告輸出
4. callback 內確認本次 run 成功後再落檔
5. 若有設定 `GCS_BUCKET_ROOT` 則寫入 GCS；否則寫本地

### 1.2 非目標

- 不做跨 run 的對話管理（session 可持續存在但不影響單次 workflow 落檔）
- 不在首版處理多 ticker 高併發 fan-out
- 不處理即時串流（streaming UI）

---

## 2. 元件與責任

### 2.1 元件

- `Cloud Scheduler`：負責定時觸發
- `Cloud Run Job`（建議）或等價容器工作負載：執行單次分析
- `GCS`：儲存最終報告

### 2.2 責任切分

- Scheduler 只負責觸發，不負責業務判斷
- Orchestrator 負責參數驗證、重試/超時、callback 落檔控制、報告持久化
- AlphaCouncil 負責分析流程本身

---

## 3. 介面契約規格

### 3.1 觸發輸入參數

必要：

- `--ticker`：股票代碼（例：`AAPL`、`2330`）

選填：

- `--market`：`us` 或 `tw`
- `--masters`：逗號分隔（例：`1,2,3`）
- `--report-format`：`json` 或 `md`

### 3.2 環境變數

- `ALPHACOUNCIL_PERSIST_ENABLED`：`true/false`，控制是否落檔
- `GCS_BUCKET_ROOT`：例如 `gs://my-bucket/alphacouncil/reports`
- `LOCAL_REPORT_ROOT`：未設定 GCS 時的本地輸出根目錄（預設 `./reports`）
- `REPORT_FORMAT`：預設輸出格式（`json|md`）

---

## 4. 完成判準與 callback 準則

### 4.1 落檔時機

在 AlphaCouncil 為固定 sequential pipeline 的前提下，最終報告由「最後一個 agent callback」觸發寫檔。

---

## 5. 報告格式與命名

### 5.1 輸出內容

最終報告建議包含：

- `meta`：`run_id`, `session_id`, `generated_at`, `ticker`, `date`, `market`, `status`
- `final_decision`：最終決策摘要

### 5.2 路徑命名

建議路徑：

`{root}/{market}/{ticker}/{date}/portfolio_report.{json|md}`

範例：

- GCS：`gs://my-bucket/alphacouncil/reports/tw/2330/2026-04-30/portfolio_report.json`
- Local：`./reports/tw/2330/2026-04-30/portfolio_report.json`

---

## 6. 排程部署規格

### 6.1 參考拓樸：Cloud Run Job + Cloud Scheduler

1. 將 orchestrator 打包成容器
2. 建立 Cloud Run Job（執行容器）
3. Cloud Scheduler 以 cron 定時觸發 Job
4. 由 Job 環境變數管理 API URL、Token、GCS root

### 6.2 時區與排程

- 時區建議固定 `Asia/Taipei`
- 台股情境可設於收盤後固定時間

---

## 7. 錯誤處理與重試規格

### 7.1 建議策略

- session 啟動失敗：直接失敗，交由雲端重試
- callback 未到達且超過 SLA：標記 timeout，exit code 非 0（可用輪詢作備援判斷）
- run failed/cancelled：exit code 非 0
- GCS 寫入失敗：exit code 非 0（避免偽成功）

### 7.2 Exit Code 建議

- `0`：成功（若啟用 persist，代表已成功落檔）
- `2`：可預期業務錯誤（HTTP/timeout/status failed）
- `3`：非預期錯誤

---

## 8. 安全與權限

- Token 不寫死在程式，使用 Secret Manager 或環境變數注入
- GCS 寫入服務帳號僅授權目標 bucket prefix
- Log 中避免輸出敏感 header/token

---

## 9. 驗收清單

- 可手動觸發一次 `ticker` 並成功完成
- 成功 session 會產生 1 份最終報告
- 失敗 session 不產生正式報告
- 未設定 `GCS_BUCKET_ROOT` 時，會正確落本地
- 設定 `GCS_BUCKET_ROOT` 時，會正確寫入 GCS
- 重跑同一日期同一 ticker 覆蓋檔案
