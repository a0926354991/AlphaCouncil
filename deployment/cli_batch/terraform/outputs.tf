output "job_name" {
  value       = google_cloud_run_v2_job.cli_batch.name
  description = "Cloud Run Job name for the CLI batch track."
}

output "workflow_name" {
  value       = google_workflows_workflow.cli_batch.name
  description = "Workflow name used to fan out tickers."
}

output "scheduler_name" {
  value       = google_cloud_scheduler_job.cli_batch.name
  description = "Cloud Scheduler job name triggering the workflow."
}

output "job_service_account" {
  value       = google_service_account.cli_batch_job.email
  description = "Service account used by the Cloud Run Job runtime."
}

output "workflow_service_account" {
  value       = google_service_account.cli_batch_workflow.email
  description = "Service account used by the Workflow runtime."
}

output "scheduler_service_account" {
  value       = google_service_account.cli_batch_scheduler.email
  description = "Service account used by Cloud Scheduler for workflow invocation."
}
