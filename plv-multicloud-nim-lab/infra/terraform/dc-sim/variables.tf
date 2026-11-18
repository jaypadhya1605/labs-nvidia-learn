variable "subscription_id" {
  type = string
}

variable "location" {
  type    = string
  default = "eastus"
}

variable "resource_group" {
  type    = string
  default = "rg-plv-dc"
}

variable "vm_size" {
  type    = string
  default = "Standard_NC24ads_A100_v4"
}

variable "priority" {
  type    = string
  default = "Spot"
}

variable "admin_ssh_public_key" {
  type        = string
  description = "Contents of ~/.ssh/id_ed25519.pub (or id_rsa.pub)."
}

variable "my_ip_cidr" {
  type        = string
  description = "Your public IP as x.x.x.x/32 - the only source allowed to SSH in."
}

variable "airgapped" {
  type        = bool
  default     = false
  description = "true = add an NSG rule that denies all outbound Internet (M9 part C). Apply after mirroring."
}
