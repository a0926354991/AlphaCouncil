output "service_name" {
  value       = google_cloud_run_v2_service.agent_service.name
  description = "Cloud Run service name for Agent Service."
}

output "service_uri" {
  value       = google_cloud_run_v2_service.agent_service.uri
  description = "Cloud Run HTTPS endpoint for Agent Service."
}

output "agent_service_account_email" {
  value       = google_service_account.agent_runtime.email
  description = "Service account email used by Agent runtime."
}

output "logs_bucket_name" {
  value       = google_storage_bucket.agent_logs.name
  description = "Telemetry bucket name passed as LOGS_BUCKET_NAME."
}

output "artifact_repository_name" {
  value       = var.artifact_repository_name
  description = "Shared Artifact Registry repository for Agent and CLI images."
}
