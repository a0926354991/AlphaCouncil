# Agent Service Deployment

本文件定義 Agent Service 主線部署方式：**Cloud Build build image + Terraform apply**。

## 1) 單一部署路徑

- Agent Service 正式入口：`make deploy-agent`
- Cloud Build 入口：`make build-agent-image`
- Terraform root：`deployment/agent_service/terraform`
- shared infra root：`deployment/shared_infra/terraform`
- Cloud Run service name 沿用 `[project].name`，本專案為 `alpha-council`

## 2) 與 CLI Batch 邊界

- Agent Service 與 CLI Batch 維持分軌
- 不共用 Terraform state
- 不共用 destroy 指令
- shared Artifact Registry repository 由 shared infra stack 管理
- Agent 與 CLI Batch 可共用同一個 Artifact Registry repository，但 image 名稱必須分開
  - Agent image：`alpha-council-agent`
  - CLI image：`alpha-council-cli`

## 3) 前置條件

- 已安裝 `gcloud`、`terraform`
- 已登入 GCP 且 active project 正確
- 已先部署 shared infra（包含 shared Artifact Registry repository 與共用 API enablement）

## 4) 主要參數

- `AGENT_SERVICE_ACCOUNT_ID`：Agent runtime SA account id
- `AGENT_LOGS_BUCKET_NAME`：bucket name only（不帶 `gs://`）
- `AGENT_IMAGE_TAG`：本次部署 image tag
- `AGENT_ALLOW_ORIGINS`：可選；多個 origin 用 `;` 分隔

`GOOGLE_CLOUD_LOCATION` 與 Cloud Run region 分離是預期設計：service 在 `asia-east1`，Vertex location 固定 `us-central1`。

## 5) 標準操作

1. Build image

```bash
make deploy-shared-infra SHARED_TF_VARS_FILE=terraform.tfvars
make build-agent-image AGENT_IMAGE_TAG=latest
```

2. Plan

```bash
make plan-agent \
  AGENT_TF_VARS_FILE=terraform.tfvars \
  AGENT_SERVICE_ACCOUNT_ID=agent-alpha-council \
  AGENT_LOGS_BUCKET_NAME=alphacouncil-agent-logs \
  AGENT_IMAGE_TAG=latest
```

3. Apply

```bash
make deploy-agent \
  AGENT_TF_VARS_FILE=terraform.tfvars \
  AGENT_SERVICE_ACCOUNT_ID=agent-alpha-council \
  AGENT_LOGS_BUCKET_NAME=alphacouncil-agent-logs \
  AGENT_IMAGE_TAG=latest
```

4. Destroy

```bash
make destroy-agent \
  AGENT_TF_VARS_FILE=terraform.tfvars \
  AGENT_SERVICE_ACCOUNT_ID=agent-alpha-council \
  AGENT_LOGS_BUCKET_NAME=alphacouncil-agent-logs \
  AGENT_IMAGE_TAG=latest
```

## 6) 驗證

- `terraform output` 檢查 service URL
- `gcloud run services describe alpha-council --region asia-east1 --project dassa-lab`
- `agents-cli run --url <service-url> --mode adk "分析 2330"`
- 驗 `/feedback` 可正常寫 log

## 7) IAM 最小權限（runtime SA）

- `roles/aiplatform.user`
- `roles/logging.logWriter`
- `roles/storage.objectAdmin`（或以 bucket policy 收斂）

若需讀取 Secret Manager，再加：

- `roles/secretmanager.secretAccessor`

## 8) Runtime 依賴提醒

- Agent 會對外讀取市場資料與新聞來源
- 若有 egress、DNS、VPC-SC 限制，需先確認可連線，避免 deploy 成功但執行階段失敗
