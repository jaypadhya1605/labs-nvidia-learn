variable "subscription_id" {
  type        = string
  description = "Azure subscription ID (az account show --query id -o tsv)."
}

variable "location" {
  type    = string
  default = "eastus"
}

variable "resource_group" {
  type    = string
  default = "rg-plv-lab"
}

variable "prefix" {
  type        = string
  default     = "plv"
  description = "Short lowercase prefix for resource names."
}

variable "kubernetes_version" {
  type        = string
  default     = null
  description = "AKS version; null = AKS default."
}

variable "system_vm_size" {
  type    = string
  default = "Standard_D4ds_v5"
}

variable "gpu_vm_size" {
  type        = string
  default     = "Standard_NC24ads_A100_v4"
  description = "1x A100 80GB PCIe. Same SKU as Labs 02/03."
}

variable "gpu_priority" {
  type    = string
  default = "Spot"
  validation {
    condition     = contains(["Spot", "Regular"], var.gpu_priority)
    error_message = "gpu_priority must be Spot or Regular."
  }
}

variable "gpu_max_nodes" {
  type        = number
  default     = 1
  description = "Cluster-autoscaler ceiling for the GPU pool. 2 for the M7 scale-out test (needs 48 vCPU of quota)."
}
