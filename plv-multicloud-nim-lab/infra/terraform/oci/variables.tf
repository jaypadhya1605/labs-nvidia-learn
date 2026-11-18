variable "region" {
  type    = string
  default = "us-ashburn-1"
}

variable "tenancy_ocid" {
  type = string
}

variable "compartment_ocid" {
  type = string
}

variable "prefix" {
  type    = string
  default = "plv"
}

variable "kubernetes_version" {
  type        = string
  default     = "v1.34.1"
  description = "Must be an OKE-supported version: oci ce cluster-options get --cluster-option-id all"
}

variable "gpu_shape" {
  type        = string
  default     = "VM.GPU.A10.1"
  description = "1x A10 24GB."
}

variable "api_allowed_cidr" {
  type        = string
  default     = "0.0.0.0/0"
  description = "Who may reach the Kubernetes API endpoint. Set to your IP/32."
}
