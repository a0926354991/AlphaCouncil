output "artifact_repository_name" {
  value       = google_artifact_registry_repository.shared.repository_id
  description = "Shared Artifact Registry repository name."
}
