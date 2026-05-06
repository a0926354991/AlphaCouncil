resource "google_service_account" "agent_runtime" {
  account_id   = var.agent_service_account_id
  display_name = "AlphaCouncil Agent Service"
}

resource "google_storage_bucket" "agent_logs" {
  name                        = var.logs_bucket_name
  project                     = var.project_id
  location                    = var.logs_bucket_location
  uniform_bucket_level_access = true
  force_destroy               = var.logs_bucket_force_destroy

  labels = var.labels
}

resource "google_project_iam_member" "agent_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.agent_runtime.email}"
}

resource "google_project_iam_member" "agent_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.agent_runtime.email}"
}

resource "google_storage_bucket_iam_member" "agent_logs_admin" {
  bucket = google_storage_bucket.agent_logs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.agent_runtime.email}"
}

resource "google_cloud_run_v2_service" "agent_service" {
  name                = var.service_name
  location            = var.region
  ingress             = var.ingress
  deletion_protection = false
  labels              = var.labels

  template {
    service_account = google_service_account.agent_runtime.email
    timeout         = format("%ss", var.timeout_seconds)

    scaling {
      min_instance_count = var.min_instance_count
      max_instance_count = var.max_instance_count
    }

    containers {
      image = var.agent_image

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "true"
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.vertex_location
      }

      env {
        name  = "LOGS_BUCKET_NAME"
        value = google_storage_bucket.agent_logs.name
      }

      env {
        name  = "ALLOW_ORIGINS"
        value = var.allow_origins
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }
    }
  }

  depends_on = [
    google_project_iam_member.agent_vertex_user,
    google_project_iam_member.agent_logging_writer,
    google_storage_bucket_iam_member.agent_logs_admin,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "agent_invoker" {
  count    = var.allow_unauthenticated ? 1 : 0
  name     = google_cloud_run_v2_service.agent_service.name
  project  = var.project_id
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
