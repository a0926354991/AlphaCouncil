.PHONY: sync lock run run-cli web api-server k8000 plan-shared-infra deploy-shared-infra destroy-shared-infra build-agent-image plan-agent deploy-agent destroy-agent build-cli-image deploy-cli-batch destroy-cli-batch check-cli-execution cleanup-gcs-reports cleanup-cloudbuild-logs cleanup-artifact-repo cleanup-all

GCP_PROJECT ?= dassa-lab
GCP_REGION ?= asia-east1
VERTEX_LOCATION ?= us-central1

SHARED_ARTIFACT_REPO ?= alpha-council
SHARED_TERRAFORM_DIR ?= deployment/shared_infra/terraform
SHARED_TF_VARS_FILE ?=

AGENT_TERRAFORM_DIR ?= deployment/agent_service/terraform
AGENT_TF_VARS_FILE ?=
AGENT_SERVICE_ACCOUNT_ID ?= agent-alpha-council
AGENT_LOGS_BUCKET_NAME ?= alphacouncil-agent-logs
AGENT_ALLOW_ORIGINS ?=
AGENT_DEPLOY_MEMORY ?= 4Gi
AGENT_DEPLOY_CPU ?= 2
AGENT_MIN_INSTANCES ?= 0
AGENT_MAX_INSTANCES ?= 3
AGENT_TIMEOUT_SECONDS ?= 900
AGENT_INGRESS ?= INGRESS_TRAFFIC_ALL
AGENT_ALLOW_UNAUTHENTICATED ?= false
AGENT_DOCKERFILE ?= Dockerfile
AGENT_CLOUDBUILD_CONFIG ?= cloudbuild.agent.yaml
AGENT_IMAGE_NAME ?= alpha-council-agent
AGENT_IMAGE_TAG ?= latest
AGENT_IMAGE_REPO ?= asia-east1-docker.pkg.dev/$(GCP_PROJECT)/$(SHARED_ARTIFACT_REPO)/$(AGENT_IMAGE_NAME)
AGENT_IMAGE_URI ?= $(AGENT_IMAGE_REPO):$(AGENT_IMAGE_TAG)

CLI_BATCH_TERRAFORM_DIR ?= deployment/cli_batch/terraform
CLI_BATCH_VARS_FILE ?=
CLI_IMAGE_NAME ?= alpha-council-cli
CLI_IMAGE_REPO ?= asia-east1-docker.pkg.dev/$(GCP_PROJECT)/$(SHARED_ARTIFACT_REPO)/$(CLI_IMAGE_NAME)
CLI_IMAGE_TAG ?= latest
CLI_IMAGE_URI ?= $(CLI_IMAGE_REPO):$(CLI_IMAGE_TAG)
CLI_DOCKERFILE ?= Dockerfile.cli
CLI_CLOUDBUILD_CONFIG ?= cloudbuild.cli.yaml
CLI_JOB_NAME ?= cli-alpha-council-job
CLI_EXECUTION_NAME ?=
CLI_GCS_OBJECT ?=

REPORTS_BUCKET_ROOT ?= gs://alphacouncil
CLOUDBUILD_LOGS_PATH ?= gs://alphacouncil/cloudbuild-logs
ARTIFACT_REPO ?=

## Install and sync Python dependencies via uv.
sync:
	uv sync

## Refresh uv lockfile from current dependency graph.
lock:
	uv lock

## Run ADK CLI flow for alpha_council agent.
run:
	uv run adk run alpha_council

## Run one full pipeline execution with local env overlays.
run-cli:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	if [ -f alpha_council/.env ]; then set -a; . ./alpha_council/.env; set +a; fi; \
	uv run alpha-council run --ticker 2330 --market tw --masters 1,2,3

## Launch ADK Web UI for local debugging.
web:
	uv run adk web

## Launch ADK API server locally.
api-server:
	uv run adk api_server

## Preview Terraform changes for shared infrastructure.
plan-shared-infra:
	@if [ ! -d "$(SHARED_TERRAFORM_DIR)" ]; then \
		echo "missing $(SHARED_TERRAFORM_DIR); add shared infra terraform before plan"; \
		exit 1; \
	fi
	@terraform -chdir="$(SHARED_TERRAFORM_DIR)" init
	@cmd='terraform -chdir="$(SHARED_TERRAFORM_DIR)" plan -var="project_id=$(GCP_PROJECT)" -var="region=$(GCP_REGION)" -var="artifact_repository_name=$(SHARED_ARTIFACT_REPO)"'; \
	if [ -n "$(SHARED_TF_VARS_FILE)" ]; then \
		cmd="$$cmd -var-file=\"$(SHARED_TF_VARS_FILE)\""; \
	fi; \
	eval "$$cmd"

## Apply Terraform changes for shared infrastructure.
deploy-shared-infra:
	@if [ ! -d "$(SHARED_TERRAFORM_DIR)" ]; then \
		echo "missing $(SHARED_TERRAFORM_DIR); add shared infra terraform before deploy"; \
		exit 1; \
	fi
	@terraform -chdir="$(SHARED_TERRAFORM_DIR)" init
	@cmd='terraform -chdir="$(SHARED_TERRAFORM_DIR)" apply -var="project_id=$(GCP_PROJECT)" -var="region=$(GCP_REGION)" -var="artifact_repository_name=$(SHARED_ARTIFACT_REPO)"'; \
	if [ -n "$(SHARED_TF_VARS_FILE)" ]; then \
		cmd="$$cmd -var-file=\"$(SHARED_TF_VARS_FILE)\""; \
	fi; \
	eval "$$cmd"

## Destroy Terraform-managed shared infrastructure.
destroy-shared-infra:
	@if [ ! -d "$(SHARED_TERRAFORM_DIR)" ]; then \
		echo "missing $(SHARED_TERRAFORM_DIR); add shared infra terraform before destroy"; \
		exit 1; \
	fi
	@terraform -chdir="$(SHARED_TERRAFORM_DIR)" init
	@cmd='terraform -chdir="$(SHARED_TERRAFORM_DIR)" destroy -var="project_id=$(GCP_PROJECT)" -var="region=$(GCP_REGION)" -var="artifact_repository_name=$(SHARED_ARTIFACT_REPO)"'; \
	if [ -n "$(SHARED_TF_VARS_FILE)" ]; then \
		cmd="$$cmd -var-file=\"$(SHARED_TF_VARS_FILE)\""; \
	fi; \
	eval "$$cmd"

## Build and push Agent Service image via Cloud Build.
build-agent-image:
	@gcloud builds submit --config "$(AGENT_CLOUDBUILD_CONFIG)" --substitutions=_AGENT_IMAGE_URI="$(AGENT_IMAGE_URI)",_AGENT_DOCKERFILE="$(AGENT_DOCKERFILE)" --gcs-log-dir="$(CLOUDBUILD_LOGS_PATH)" --project "$(GCP_PROJECT)" .

## Preview Terraform changes for Agent Service deployment.
plan-agent:
	@if [ ! -d "$(AGENT_TERRAFORM_DIR)" ]; then \
		echo "missing $(AGENT_TERRAFORM_DIR); add agent terraform before plan"; \
		exit 1; \
	fi
	@terraform -chdir="$(AGENT_TERRAFORM_DIR)" init
	@cmd='terraform -chdir="$(AGENT_TERRAFORM_DIR)" plan -var="project_id=$(GCP_PROJECT)" -var="region=$(GCP_REGION)" -var="vertex_location=$(VERTEX_LOCATION)" -var="artifact_repository_name=$(SHARED_ARTIFACT_REPO)" -var="agent_image=$(AGENT_IMAGE_URI)" -var="agent_service_account_id=$(AGENT_SERVICE_ACCOUNT_ID)" -var="logs_bucket_name=$(AGENT_LOGS_BUCKET_NAME)" -var="allow_origins=$(AGENT_ALLOW_ORIGINS)" -var="memory=$(AGENT_DEPLOY_MEMORY)" -var="cpu=$(AGENT_DEPLOY_CPU)" -var="min_instance_count=$(AGENT_MIN_INSTANCES)" -var="max_instance_count=$(AGENT_MAX_INSTANCES)" -var="timeout_seconds=$(AGENT_TIMEOUT_SECONDS)" -var="ingress=$(AGENT_INGRESS)" -var="allow_unauthenticated=$(AGENT_ALLOW_UNAUTHENTICATED)"'; \
	if [ -n "$(AGENT_TF_VARS_FILE)" ]; then \
		cmd="$$cmd -var-file=\"$(AGENT_TF_VARS_FILE)\""; \
	fi; \
	eval "$$cmd"

## Apply Terraform changes for Agent Service deployment.
deploy-agent:
	@if [ ! -d "$(AGENT_TERRAFORM_DIR)" ]; then \
		echo "missing $(AGENT_TERRAFORM_DIR); add agent terraform before deploy"; \
		exit 1; \
	fi
	@terraform -chdir="$(AGENT_TERRAFORM_DIR)" init
	@cmd='terraform -chdir="$(AGENT_TERRAFORM_DIR)" apply -var="project_id=$(GCP_PROJECT)" -var="region=$(GCP_REGION)" -var="vertex_location=$(VERTEX_LOCATION)" -var="artifact_repository_name=$(SHARED_ARTIFACT_REPO)" -var="agent_image=$(AGENT_IMAGE_URI)" -var="agent_service_account_id=$(AGENT_SERVICE_ACCOUNT_ID)" -var="logs_bucket_name=$(AGENT_LOGS_BUCKET_NAME)" -var="allow_origins=$(AGENT_ALLOW_ORIGINS)" -var="memory=$(AGENT_DEPLOY_MEMORY)" -var="cpu=$(AGENT_DEPLOY_CPU)" -var="min_instance_count=$(AGENT_MIN_INSTANCES)" -var="max_instance_count=$(AGENT_MAX_INSTANCES)" -var="timeout_seconds=$(AGENT_TIMEOUT_SECONDS)" -var="ingress=$(AGENT_INGRESS)" -var="allow_unauthenticated=$(AGENT_ALLOW_UNAUTHENTICATED)"'; \
	if [ -n "$(AGENT_TF_VARS_FILE)" ]; then \
		cmd="$$cmd -var-file=\"$(AGENT_TF_VARS_FILE)\""; \
	fi; \
	eval "$$cmd"

## Destroy Terraform-managed Agent Service resources.
destroy-agent:
	@if [ ! -d "$(AGENT_TERRAFORM_DIR)" ]; then \
		echo "missing $(AGENT_TERRAFORM_DIR); add agent terraform before destroy"; \
		exit 1; \
	fi
	@terraform -chdir="$(AGENT_TERRAFORM_DIR)" init
	@cmd='terraform -chdir="$(AGENT_TERRAFORM_DIR)" destroy -var="project_id=$(GCP_PROJECT)" -var="region=$(GCP_REGION)" -var="vertex_location=$(VERTEX_LOCATION)" -var="artifact_repository_name=$(SHARED_ARTIFACT_REPO)" -var="agent_image=$(AGENT_IMAGE_URI)" -var="agent_service_account_id=$(AGENT_SERVICE_ACCOUNT_ID)" -var="logs_bucket_name=$(AGENT_LOGS_BUCKET_NAME)" -var="allow_origins=$(AGENT_ALLOW_ORIGINS)" -var="memory=$(AGENT_DEPLOY_MEMORY)" -var="cpu=$(AGENT_DEPLOY_CPU)" -var="min_instance_count=$(AGENT_MIN_INSTANCES)" -var="max_instance_count=$(AGENT_MAX_INSTANCES)" -var="timeout_seconds=$(AGENT_TIMEOUT_SECONDS)" -var="ingress=$(AGENT_INGRESS)" -var="allow_unauthenticated=$(AGENT_ALLOW_UNAUTHENTICATED)"'; \
	if [ -n "$(AGENT_TF_VARS_FILE)" ]; then \
		cmd="$$cmd -var-file=\"$(AGENT_TF_VARS_FILE)\""; \
	fi; \
	eval "$$cmd"

## Build and push CLI Batch image via Cloud Build.
build-cli-image:
	@gcloud builds submit --config "$(CLI_CLOUDBUILD_CONFIG)" --substitutions=_CLI_IMAGE_URI="$(CLI_IMAGE_URI)",_CLI_DOCKERFILE="$(CLI_DOCKERFILE)" --gcs-log-dir="$(CLOUDBUILD_LOGS_PATH)" --project "$(GCP_PROJECT)" .

## Apply Terraform changes for CLI Batch resources.
deploy-cli-batch:
	@if [ ! -d "$(CLI_BATCH_TERRAFORM_DIR)" ]; then \
		echo "missing $(CLI_BATCH_TERRAFORM_DIR); add CLI batch terraform before deploy"; \
		exit 1; \
	fi
	@terraform -chdir="$(CLI_BATCH_TERRAFORM_DIR)" init
	@if [ -z "$(CLI_IMAGE_URI)" ]; then \
		echo "set CLI_IMAGE_URI before deploy-cli-batch"; \
		exit 1; \
	fi
	@if [ -n "$(CLI_BATCH_VARS_FILE)" ]; then \
		terraform -chdir="$(CLI_BATCH_TERRAFORM_DIR)" apply -var="cli_image=$(CLI_IMAGE_URI)" -var-file="$(CLI_BATCH_VARS_FILE)"; \
	else \
		terraform -chdir="$(CLI_BATCH_TERRAFORM_DIR)" apply -var="cli_image=$(CLI_IMAGE_URI)"; \
	fi

## Destroy Terraform-managed CLI Batch resources.
destroy-cli-batch:
	@if [ ! -d "$(CLI_BATCH_TERRAFORM_DIR)" ]; then \
		echo "missing $(CLI_BATCH_TERRAFORM_DIR); add CLI batch terraform before destroy"; \
		exit 1; \
	fi
	@terraform -chdir="$(CLI_BATCH_TERRAFORM_DIR)" init
	@if [ -z "$(CLI_IMAGE_URI)" ]; then \
		echo "set CLI_IMAGE_URI before destroy-cli-batch"; \
		exit 1; \
	fi
	@if [ -n "$(CLI_BATCH_VARS_FILE)" ]; then \
		terraform -chdir="$(CLI_BATCH_TERRAFORM_DIR)" destroy -var="cli_image=$(CLI_IMAGE_URI)" -var-file="$(CLI_BATCH_VARS_FILE)"; \
	else \
		terraform -chdir="$(CLI_BATCH_TERRAFORM_DIR)" destroy -var="cli_image=$(CLI_IMAGE_URI)"; \
	fi

## Check Cloud Run Job execution and optional report artifact.
check-cli-execution:
	@if [ -z "$(CLI_EXECUTION_NAME)" ]; then \
		echo "set CLI_EXECUTION_NAME before check-cli-execution"; \
		exit 1; \
	fi
	@cmd='python scripts/check_cli_batch_execution.py "$(CLI_EXECUTION_NAME)" --job-name "$(CLI_JOB_NAME)" --region "$(GCP_REGION)" --project "$(GCP_PROJECT)"'; \
	if [ -n "$(CLI_GCS_OBJECT)" ]; then \
		cmd="$$cmd --gcs-object \"$(CLI_GCS_OBJECT)\""; \
	fi; \
	eval "$$cmd"

## Remove generated report objects under configured GCS root.
cleanup-gcs-reports:
	@if [ -z "$(REPORTS_BUCKET_ROOT)" ]; then \
		echo "set REPORTS_BUCKET_ROOT to the reports path to remove"; \
		exit 1; \
	fi
	gcloud storage rm --recursive "$(REPORTS_BUCKET_ROOT)/**"

## Remove Cloud Build logs under configured GCS path.
cleanup-cloudbuild-logs:
	@if [ -z "$(CLOUDBUILD_LOGS_PATH)" ]; then \
		echo "set CLOUDBUILD_LOGS_PATH to the Cloud Build log path to remove"; \
		exit 1; \
	fi
	gcloud storage rm --recursive "$(CLOUDBUILD_LOGS_PATH)/**"

## Delete Artifact Registry repository by name.
cleanup-artifact-repo:
	@if [ -z "$(ARTIFACT_REPO)" ]; then \
		echo "set ARTIFACT_REPO to the Artifact Registry repository name to delete"; \
		exit 1; \
	fi
	gcloud artifacts repositories delete "$(ARTIFACT_REPO)" --location "$(GCP_REGION)" --project "$(GCP_PROJECT)"

cleanup-all: cleanup-gcs-reports cleanup-cloudbuild-logs cleanup-artifact-repo

## Kill local processes listening on port 8000.
k8000:
	@pids="$$(lsof -ti :8000)"; \
	if [ -n "$$pids" ]; then \
		kill -9 $$pids; \
		echo "Killed process(es) on :8000 -> $$pids"; \
	else \
		echo "No process is listening on :8000"; \
	fi
