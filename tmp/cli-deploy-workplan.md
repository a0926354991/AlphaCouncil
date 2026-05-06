# CLI Deploy Workplan (Scaffold + CLI Batch Split)

Date: 2026-05-05
Branch: `exp/cloud-run-cli-backtest-scaffold`

## Phase 0 - Decision Freeze

- Keep dual-track deployment:
  - Agent Service (agents-cli scaffold standard path)
  - CLI Batch Experiment (Cloud Run Job + Workflows + Scheduler)
- Keep fixed experiment config:
  - project: `dassa-lab`
  - region: `asia-east1`
  - vertex location: `us-central1`
  - schedule: `10 16 * * 1-5`
  - timezone: `Asia/Taipei`
  - bucket root: `gs://alphacouncil`
  - masters: `1,2,3`
  - report format: `json`
  - timeout: `1800`
  - tickers:
    - `2330,2308,2454,2317,3711,2383,2345,3037,2303,2382,2881,2891,2882,2886,2327`

## Phase 1 - Scaffold Output Review (Agent Track)

- Review enhance outputs created with:
  - `agents-cli scaffold enhance . --deployment-target cloud_run --agent-directory alpha_council`
- Validate and keep/remove:
  - `Dockerfile`
  - `alpha_council/app_utils/`
  - `alpha_council/fast_api_app.py`
  - `tests/eval/`, `tests/integration/`, `tests/unit/`
- Audit `alpha_council/agent.py` auto changes (`app` compatibility object)
- Produce keep/remove rationale note in docs

## Phase 2 - Infra Split and State Isolation

- Ensure Terraform root/state separation:
  - Agent root/state
  - CLI Batch root/state
- Ensure naming and IAM isolation:
  - `agent-*` vs `cli-*`
  - dedicated service accounts per track
- Ensure no shared destroy blast radius

## Phase 3 - CLI Batch Scaffold-Style Integration

- Place CLI batch infra in dedicated path aligned with repo deployment style
- Include resources:
  - Cloud Run Job
  - Workflows (parallel fan-out)
  - Scheduler (weekday 16:10)
  - SA + IAM minimum roles
- Keep CLI invocation unchanged:
  - `alpha-council run --ticker ... --market tw --masters 1,2,3 --report-format json --timeout-seconds 1800`
- Keep VPC-SC-safe build logs:
  - `--gcs-log-dir=gs://alphacouncil/cloudbuild-logs`

## Phase 4 - Unified Operator Entrypoints

- Keep Makefile thin and explicit:
  - `deploy-agent`
  - `deploy-cli-batch`
  - `destroy-cli-batch`
  - `cleanup-gcs-reports`
  - `cleanup-cloudbuild-logs`
  - `cleanup-artifact-repo`
  - `cleanup-all`

## Phase 5 - Validation Matrix

- Deployment checks:
  - build image success
  - terraform apply success
- Runtime checks:
  - workflow run with 1 ticker success
  - workflow run with 15 tickers success
  - GCS object path exists: `gs://alphacouncil/tw/<ticker>/<date>/portfolio_report.json`
- Teardown checks:
  - terraform destroy removes managed resources
  - non-terraform residues removable via cleanup scripts

## Phase 6 - Docs and Handover

- Update docs pointers and command examples
- Ensure ADR reflects final directory + deployment policy
- Split commits into logical units:
  1. scaffold outputs
  2. cli-batch infra integration
  3. docs + make + cleanup

## Risks and Mitigations

- VPC-SC blocks Cloud Build default logs
  - Mitigation: always set `--gcs-log-dir`
- Vertex model not available in `asia-east1`
  - Mitigation: set `GOOGLE_CLOUD_LOCATION=us-central1` for job runtime
- Accidental cross-track delete
  - Mitigation: separate tf state and explicit destroy targets

## Completion Criteria

- Dual-track structure is clear and documented
- Agent deploy path follows agents-cli scaffold style
- CLI batch deploy path is independent and reproducible
- Deploy, run, persist, destroy, and cleanup are all verified
