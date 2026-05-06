# Agent Service Terraform

這個目錄只管理 Agent Service 主線資源：

- Cloud Run service (`alpha-council`)
- Agent runtime service account
- Agent telemetry bucket
- Agent 專用 IAM 綁定
- 必要 API enablement

## 與其他 Terraform root 的邊界

- shared Artifact Registry repository 由 `deployment/shared_infra/terraform` 管理
- CLI Batch Terraform root 維持在 `deployment/cli_batch/terraform`
- 這裡不管理 CLI Batch 的 Job/Workflow/Scheduler

## 使用方式

1. 建立 tfvars：

```bash
cp deployment/agent_service/terraform/terraform.tfvars.example deployment/agent_service/terraform/terraform.tfvars
```

2. 建置 Agent image（Cloud Build）：

```bash
make build-agent-image AGENT_IMAGE_TAG=<tag>
```

3. 預覽變更：

```bash
make plan-agent \
  AGENT_TF_VARS_FILE=terraform.tfvars \
  AGENT_IMAGE_TAG=<tag>
```

4. 部署：

```bash
make deploy-agent \
  AGENT_TF_VARS_FILE=terraform.tfvars \
  AGENT_IMAGE_TAG=<tag>
```

5. 銷毀：

```bash
make destroy-agent \
  AGENT_TF_VARS_FILE=terraform.tfvars \
  AGENT_IMAGE_TAG=<tag>
```

## 注意

- shared Artifact Registry repository 不在此 stack 建立或刪除
- `logs_bucket_name` 一律為 bucket name，不帶 `gs://`
- `logs_bucket_force_destroy` 預設 `false`，可避免誤刪有資料的 bucket
