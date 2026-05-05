# CLI Batch Experiment Deployment

本文件定義 CLI 實驗部署（非主線服務）做法，與 Agent Service 分離。

## 1) 實驗目標

- 固定時間觸發分析
- 一次傳入多個股票代號並行執行
- 結果寫入 GCS，累積 n 天資料供回測

## 2) 固定參數（目前決議）

- project: `dassa-lab`
- region: `asia-east1`
- vertex model location: `us-central1`
- schedule: `10 16 * * 1-5`
- timezone: `Asia/Taipei`
- bucket root: `gs://alphacouncil`
- masters: `1,2,3`
- report format: `json`
- timeout seconds: `1800`
- tickers:
  - `2330,2308,2454,2317,3711,2383,2345,3037,2303,2382,2881,2891,2882,2886,2327`

## 3) 執行拓樸

- Cloud Scheduler：定時觸發
- Cloud Workflows：並行 fan-out（多 ticker）
- Cloud Run Job：執行 `alpha-council run`
- GCS：持久化報告

## 4) CLI 介面對齊原則

完全沿用現有 CLI 參數，不新增 deployment 專用 flag：

- `--ticker`
- `--market`
- `--masters`
- `--report-format`
- `--timeout-seconds`

Job 透過 env 控制持久化：

- `ALPHACOUNCIL_PERSIST_ENABLED=true`
- `GCS_BUCKET_ROOT=gs://alphacouncil`
- `GOOGLE_GENAI_USE_VERTEXAI=true`
- `GOOGLE_CLOUD_PROJECT=dassa-lab`
- `GOOGLE_CLOUD_LOCATION=us-central1`

## 5) 權限與安全

Job service account 建議最小權限：

- `roles/storage.objectCreator`
- `roles/logging.logWriter`
- `roles/aiplatform.user`

Workflow/Scheduler 僅授予觸發所需權限，不使用廣域管理角色。

## 6) VPC-SC 注意事項

若組織有 VPC Service Controls，Cloud Build 可能因預設 logs bucket 跨 perimeter 被擋。

建議固定使用：

`--gcs-log-dir=gs://alphacouncil/cloudbuild-logs`

## 7) 清理策略

- Terraform destroy：只刪 Terraform 管理資源（Job/Workflow/Scheduler/SA/IAM）
- 非 Terraform 殘留需另清：
  - GCS reports
  - Cloud Build logs
  - Artifact Registry repository/image

## 8) 實驗分支策略

- 建議在實驗分支執行與驗證
- 不直接影響 main runtime path
- 驗證穩定後再挑選可產品化部分回主線
