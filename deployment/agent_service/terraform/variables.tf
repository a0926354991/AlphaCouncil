variable "project_id" {
  description = "GCP project ID for Agent Service resources."
  type        = string
  default     = "dassa-lab"
}

variable "region" {
  description = "GCP region for Agent Service Cloud Run deployment."
  type        = string
  default     = "asia-east1"
}

variable "vertex_location" {
  description = "Vertex AI location used by the runtime."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name for Agent Service."
  type        = string
  default     = "alpha-council"
}

variable "artifact_repository_name" {
  description = "Shared Artifact Registry repository name used by Agent and CLI images."
  type        = string
  default     = "alpha-council"
}

variable "agent_image" {
  description = "Container image URI for Agent Service Cloud Run deployment."
  type        = string
}

variable "agent_service_account_id" {
  description = "Service account ID created for Agent runtime."
  type        = string
  default     = "agent-alpha-council"
}

variable "logs_bucket_name" {
  description = "Telemetry bucket name only (without gs://)."
  type        = string
  default     = "alphacouncil-agent-logs"
}

variable "logs_bucket_location" {
  description = "Bucket location for Agent telemetry bucket."
  type        = string
  default     = "ASIA-EAST1"
}

variable "logs_bucket_force_destroy" {
  description = "Allow deleting non-empty logs bucket on destroy."
  type        = bool
  default     = false
}

variable "allow_origins" {
  description = "Optional semicolon-separated origin list for CORS."
  type        = string
  default     = ""
}

variable "memory" {
  description = "Cloud Run memory limit."
  type        = string
  default     = "4Gi"
}

variable "cpu" {
  description = "Cloud Run CPU limit."
  type        = string
  default     = "2"
}

variable "min_instance_count" {
  description = "Minimum Cloud Run instances."
  type        = number
  default     = 0
}

variable "max_instance_count" {
  description = "Maximum Cloud Run instances."
  type        = number
  default     = 3
}

variable "timeout_seconds" {
  description = "Cloud Run request timeout in seconds."
  type        = number
  default     = 900
}

variable "ingress" {
  description = "Cloud Run ingress setting."
  type        = string
  default     = "INGRESS_TRAFFIC_ALL"

  validation {
    condition = contains([
      "INGRESS_TRAFFIC_ALL",
      "INGRESS_TRAFFIC_INTERNAL_ONLY",
      "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER",
    ], var.ingress)
    error_message = "ingress must be one of INGRESS_TRAFFIC_ALL, INGRESS_TRAFFIC_INTERNAL_ONLY, INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER."
  }
}

variable "allow_unauthenticated" {
  description = "Allow unauthenticated invocation of Cloud Run service."
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels applied to managed resources where supported."
  type        = map(string)
  default = {
    managed-by = "terraform"
    track      = "agent-service"
  }
}
