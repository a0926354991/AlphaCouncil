# ADR-0001: Split Agent Service and CLI Batch Deployment

- Status: Accepted
- Date: 2026-05-05

## Context

專案同時存在兩種執行模型：

- 一般 Agent 服務（長駐或互動型）
- CLI 批次實驗（定時、可並行、可重跑）

`agents-cli scaffold/enhance` 產生的是一般 agent deployment 結構，與 CLI 批次回測需求不完全重疊。

## Decision

採用雙軌部署：

1. Agent Service：沿用 `agents-cli` 標準部署結構
2. CLI Batch：建立獨立部署切面（Cloud Run Job + Workflows + Scheduler）

並強制以下分離原則：

- Terraform state 分離
- Service Account 分離
- 命名空間分離
- IAM 權限分離

## Consequences

正面：

- 降低相互影響風險，回滾更安全
- 批次實驗可快速迭代，不污染主線服務
- destroy 與 cleanup 可以精準控制

負面：

- 維護兩套部署流程
- 文件與命名規範需要更嚴謹

## Alternatives Considered

- 單一 Terraform root 同時管理 Agent + CLI Batch
  - 未採用，因 blast radius 較大、實驗變更容易影響主線

## Notes

短期以「可驗證、可回收」為優先；若 CLI Batch 長期產品化，再評估與主線做更高層級模組化整合。
