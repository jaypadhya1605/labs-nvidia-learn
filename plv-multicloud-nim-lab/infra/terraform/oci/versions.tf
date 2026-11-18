terraform {
  required_version = ">= 1.9"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 9.10"
    }
  }
}

# Authentication comes from ~/.oci/config (oci setup config).
provider "oci" {
  region = var.region
}
