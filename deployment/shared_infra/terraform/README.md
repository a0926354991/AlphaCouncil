# Shared Infra Terraform

這個目錄只管理 AlphaCouncil 的 shared infrastructure：

- 共用 API enablement（Run / Cloud Build / Artifact Registry / Vertex / Storage）
- shared Artifact Registry repository

## 與其他 Terraform root 的邊界

- Agent Service 專屬資源在 `deployment/agent_service/terraform`
- CLI Batch 專屬資源在 `deployment/cli_batch/terraform`
- 這裡只管理會被兩條線共用的資源 owner

## 使用方式

1. 建立 tfvars：

```bash
cp deployment/shared_infra/terraform/terraform.tfvars.example deployment/shared_infra/terraform/terraform.tfvars
```

2. 預覽變更：

```bash
make plan-shared-infra SHARED_TF_VARS_FILE=terraform.tfvars
```

3. 部署：

```bash
make deploy-shared-infra SHARED_TF_VARS_FILE=terraform.tfvars
```

4. 銷毀：

```bash
make destroy-shared-infra SHARED_TF_VARS_FILE=terraform.tfvars
```

## 注意

- Agent 與 CLI image 都放在同一個 repository
- image name 必須分開管理：
  - Agent: `alpha-council-agent`
  - CLI: `alpha-council-cli`
