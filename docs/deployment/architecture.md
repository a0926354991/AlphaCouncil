# Deployment Architecture

本專案採用雙軌部署架構，將一般 Agent 服務與 CLI 批次實驗解耦，避免互相影響。

## 1) 兩塊部署邊界

- Shared Infra
  - 目標：管理 Agent 與 CLI Batch 共同依賴的 shared resources
  - 入口：`make deploy-shared-infra`
- Agent Service（主線）
  - 目標：對外服務、一般 agent runtime
  - 來源：Cloud Build image + Terraform Cloud Run service
  - 入口：`make deploy-agent`（Cloud Build + Terraform）
- CLI Batch（實驗）
  - 目標：排程批次實驗、回測資料產生
  - 拓樸：`Cloud Scheduler -> Cloud Workflows -> Cloud Run Job -> GCS`

## 2) 解耦規則

- Terraform root/state 必須分開（不可共用同一個 state）
- Service Account 分開（Agent 與 CLI Batch 不共用）
- 資源命名分開（建議 `agent-*` 與 `cli-*` 前綴）
- IAM 最小權限，不跨線授權

## 3) 共用設定

- GCP project 可以共用，但 deployment path 要分離
- 可共用環境參數：`project_id`, `region`, `labels`
- 共用觀測能力（Logging/Monitoring）但維持資源隔離

## 4) 操作策略

- `deploy-agent` 與 `deploy-cli-batch` 分開執行
- `deploy-shared-infra`、`deploy-agent`、`deploy-cli-batch` 分開執行
- `destroy` 分開執行，避免誤刪另一條線資源
- 批次實驗清理（reports/logs/images）視資料保留政策決定
- Agent Service 名稱直接沿用 `[project].name = "alpha-council"`
- Agent 與 CLI Batch 可共用 Artifact Registry repository，但 image name 與 Terraform owner 必須分離

## 5) 失敗隔離

- CLI Batch 模型權限、排程錯誤、資料寫入錯誤，不應影響 Agent Service
- Agent Service 升版或回滾，不應改動 CLI Batch 的排程與工作流
