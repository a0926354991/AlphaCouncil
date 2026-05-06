variable "project_id" {
  description = "GCP project ID for shared AlphaCouncil infrastructure."
  type        = string
  default     = "dassa-lab"
}

variable "region" {
  description = "GCP region for shared AlphaCouncil infrastructure."
  type        = string
  default     = "asia-east1"
}

variable "artifact_repository_name" {
  description = "Shared Artifact Registry repository used by Agent and CLI images."
  type        = string
  default     = "alpha-council"
}

variable "labels" {
  description = "Labels applied to shared resources where supported."
  type        = map(string)
  default = {
    managed-by = "terraform"
    track      = "shared-infra"
  }
}
