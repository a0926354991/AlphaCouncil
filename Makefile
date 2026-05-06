.PHONY: sync lock run run-cli web api-server k8000 build-cli-image deploy-agent deploy-cli-batch destroy-cli-batch check-cli-execution cleanup-gcs-reports cleanup-cloudbuild-logs cleanup-artifact-repo cleanup-all

GCP_PROJECT ?= dassa-lab
GCP_REGION ?= asia-east1
VERTEX_LOCATION ?= us-central1

AGENT_SERVICE_ACCOUNT ?=
AGENT_LOGS_BUCKET ?=
AGENT_ALLOW_ORIGINS ?=
AGENT_DEPLOY_MEMORY ?= 4Gi

CLI_BATCH_TERRAFORM_DIR ?= deployment/cli_batch/terraform
CLI_BATCH_VARS_FILE ?=
CLI_IMAGE_REPO ?= asia-east1-docker.pkg.dev/$(GCP_PROJECT)/cli-alpha-council/alpha-council
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

sync:
	uv sync

lock:
	uv lock

run:
	uv run adk run alpha_council

run-cli:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	if [ -f alpha_council/.env ]; then set -a; . ./alpha_council/.env; set +a; fi; \
	uv run alpha-council run --ticker 2330 --market tw --masters 1,2,3

web:
	uv run adk web

api-server:
	uv run adk api_server

deploy-agent:
	@if [ -z "$(AGENT_SERVICE_ACCOUNT)" ]; then \
		echo "set AGENT_SERVICE_ACCOUNT before deploy-agent"; \
		exit 1; \
	fi
	@if [ -z "$(AGENT_LOGS_BUCKET)" ]; then \
		echo "set AGENT_LOGS_BUCKET before deploy-agent"; \
		exit 1; \
	fi
	@env_vars="^@^GOOGLE_GENAI_USE_VERTEXAI=true@GOOGLE_CLOUD_PROJECT=$(GCP_PROJECT)@GOOGLE_CLOUD_LOCATION=$(VERTEX_LOCATION)@LOGS_BUCKET_NAME=$(AGENT_LOGS_BUCKET)@ALLOW_ORIGINS=$(AGENT_ALLOW_ORIGINS)"; \
	cmd='agents-cli deploy --project "$(GCP_PROJECT)" --region "$(GCP_REGION)" --port 8080 --memory "$(AGENT_DEPLOY_MEMORY)" --update-env-vars "'"$$env_vars"'" --no-confirm-project'; \
	cmd="$$cmd --service-account \"$(AGENT_SERVICE_ACCOUNT)\""; \
	eval "$$cmd"

build-cli-image:
	@gcloud builds submit --config "$(CLI_CLOUDBUILD_CONFIG)" --substitutions=_CLI_IMAGE_URI="$(CLI_IMAGE_URI)",_CLI_DOCKERFILE="$(CLI_DOCKERFILE)" --gcs-log-dir="$(CLOUDBUILD_LOGS_PATH)" --project "$(GCP_PROJECT)" .

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

cleanup-gcs-reports:
	@if [ -z "$(REPORTS_BUCKET_ROOT)" ]; then \
		echo "set REPORTS_BUCKET_ROOT to the reports path to remove"; \
		exit 1; \
	fi
	gcloud storage rm --recursive "$(REPORTS_BUCKET_ROOT)/**"

cleanup-cloudbuild-logs:
	@if [ -z "$(CLOUDBUILD_LOGS_PATH)" ]; then \
		echo "set CLOUDBUILD_LOGS_PATH to the Cloud Build log path to remove"; \
		exit 1; \
	fi
	gcloud storage rm --recursive "$(CLOUDBUILD_LOGS_PATH)/**"

cleanup-artifact-repo:
	@if [ -z "$(ARTIFACT_REPO)" ]; then \
		echo "set ARTIFACT_REPO to the Artifact Registry repository name to delete"; \
		exit 1; \
	fi
	gcloud artifacts repositories delete "$(ARTIFACT_REPO)" --location "$(GCP_REGION)" --project "$(GCP_PROJECT)"

cleanup-all: cleanup-gcs-reports cleanup-cloudbuild-logs cleanup-artifact-repo

k8000:
	@pids="$$(lsof -ti :8000)"; \
	if [ -n "$$pids" ]; then \
		kill -9 $$pids; \
		echo "Killed process(es) on :8000 -> $$pids"; \
	else \
		echo "No process is listening on :8000"; \
	fi
