variable "project" {
  type        = string
  description = "GCP project ID."
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type        = string
  default     = "us-central1-a"
  description = "Zonal cluster (covered by the GKE free-tier management credit). Must offer your GPU."
}

variable "prefix" {
  type    = string
  default = "plv"
}

variable "gpu_profile" {
  type        = string
  default     = "rtxpro6000"
  description = "rtxpro6000 (G4, RTX PRO 6000 Blackwell 96GB) | a100-40 (A2, A100 40GB) | l4 (G2, L4 24GB)."
  validation {
    condition     = contains(["rtxpro6000", "a100-40", "l4"], var.gpu_profile)
    error_message = "gpu_profile must be rtxpro6000, a100-40 or l4."
  }
}

variable "gpu_spot" {
  type    = bool
  default = true
}

variable "gpu_max_nodes" {
  type    = number
  default = 2
}
