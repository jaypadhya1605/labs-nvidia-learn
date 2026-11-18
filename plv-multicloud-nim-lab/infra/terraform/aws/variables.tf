variable "region" {
  type    = string
  default = "us-east-1"
}

variable "prefix" {
  type    = string
  default = "plv"
}

variable "kubernetes_version" {
  type        = string
  default     = null
  description = "EKS version; null = EKS default."
}

variable "gpu_instance_type" {
  type        = string
  default     = "g6e.xlarge"
  description = "1x L40S 48GB. g6.2xlarge = 1x L4 (cheaper, not on the NIM verified list for Llama 3.1 8B)."
}

variable "gpu_spot" {
  type    = bool
  default = true
}

variable "gpu_max_nodes" {
  type    = number
  default = 2
}
