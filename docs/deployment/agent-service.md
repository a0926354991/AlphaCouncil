# Agent Service Deployment

本文件定義 Agent Service 主線部署方式。這條線以 `agents-cli deploy` 為唯一標準入口，不與 CLI Batch 混用。

## 1) 單一設定來源

- `pyproject.toml`
  - `[tool.agents-cli].agent_directory = "alpha_council"`
  - `[tool.agents-cli].region = "asia-east1"`
  - `[tool.agents-cli.create_params].deployment_target = "cloud_run"`
- Cloud Run service name 直接取 `[project].name`，本專案會部署為 `alpha-council`

## 2) 前置條件

- 已安裝 `agents-cli` 與 `gcloud`
- 已登入 GCP，且 `dassa-lab` 有部署權限
- 已啟用 API:
  - `cloudbuild.googleapis.com`
  - `run.googleapis.com`
- 建議一併啟用：
  - `aiplatform.googleapis.com`
  - `artifactregistry.googleapis.com`
  - `storage.googleapis.com`
- 已準備 Agent 專用 service account
- 已準備 Agent telemetry bucket，並將 bucket 名稱提供給 `LOGS_BUCKET_NAME`

## 3) 建議環境變數

- `GOOGLE_GENAI_USE_VERTEXAI=true`
- `GOOGLE_CLOUD_PROJECT=dassa-lab`
- `GOOGLE_CLOUD_LOCATION=us-central1`
- `LOGS_BUCKET_NAME=<agent-logs-bucket>`
- `ALLOW_ORIGINS=<comma-separated-origins>`

`GOOGLE_CLOUD_LOCATION` 與 Cloud Run region 分離是預期設計：service 佈在 `asia-east1`，Vertex model location 固定 `us-central1`。

## 4) 標準部署指令

優先使用 `Makefile` 包裝後的標準入口：

```bash
make deploy-agent \
  AGENT_SERVICE_ACCOUNT=agent-alpha-council@dassa-lab.iam.gserviceaccount.com \
  AGENT_LOGS_BUCKET=alphacouncil-agent-logs \
  AGENT_ALLOW_ORIGINS=https://app.example.com,https://admin.example.com
```

對應的底層行為是：

```bash
agents-cli deploy \
  --project dassa-lab \
  --region asia-east1 \
  --port 8080 \
  --memory 4Gi \
  --service-account <agent-sa> \
  --update-env-vars '^@^GOOGLE_GENAI_USE_VERTEXAI=true@GOOGLE_CLOUD_PROJECT=dassa-lab@GOOGLE_CLOUD_LOCATION=us-central1@LOGS_BUCKET_NAME=<bucket>@ALLOW_ORIGINS=<origins>' \
  --no-confirm-project
```

這裡使用 `^@^...@...` delimiter syntax，避免 `ALLOW_ORIGINS` 內部的逗號被 `gcloud` 誤判成不同 env var。

## 5) 驗證

- 設定 dry run：

```bash
agents-cli deploy --dry-run --project dassa-lab --region asia-east1 --no-confirm-project
```

- 服務狀態：

```bash
agents-cli deploy --status --project dassa-lab --region asia-east1
```

- Smoke test:
  - `gcloud run services describe alpha-council --region asia-east1 --project dassa-lab`
  - 取得 service URL 後，用 `agents-cli run --url <service-url> --mode adk "分析 2330"`
  - 驗 `/feedback` 可成功寫入 log

## 6) IAM 最小權限

Agent service account 至少需要：

- `roles/aiplatform.user`
- `roles/logging.logWriter`
- `roles/storage.objectAdmin` 或依 bucket policy 收斂到更小範圍

如果 Agent 需要讀取 Secret Manager，再額外授予：

- `roles/secretmanager.secretAccessor`

## 7) 回滾

Cloud Run service 可直接用 revision traffic rollback：

```bash
gcloud run revisions list --service=alpha-council --region=asia-east1 --project=dassa-lab
gcloud run services update-traffic alpha-council --to-revisions=<revision>=100 --region=asia-east1 --project=dassa-lab
```

若是設定或程式問題，仍以修正後重新 `make deploy-agent` 為主。

## 8) 與 CLI Batch 的邊界

- `agents-cli deploy` 只負責 Agent Service
- CLI Batch 的 `Cloud Run Job + Workflows + Scheduler` 必須維持獨立部署切面
- 不共用 destroy 指令、不共用 service account、不共用 Terraform state
