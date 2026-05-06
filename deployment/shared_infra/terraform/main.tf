locals {
  required_services = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
  ])
}

resource "google_project_service" "shared_services" {
  for_each           = local.required_services
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "shared" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_repository_name
  format        = "DOCKER"
  description   = "Shared repo for AlphaCouncil Agent and CLI images"
  labels        = var.labels

  depends_on = [google_project_service.shared_services]
}
