# Plover Lab 04 - Oracle Cloud side (stretch): VCN, enhanced OKE cluster (needed for OKE workload identity),
# a CPU pool and a GPU pool on Oracle Linux GPU images (driver preinstalled), OCI Vault secrets that External
# Secrets reads with workload identity. OKE node pools have no taint field - infra/tf.sh taints GPU nodes.

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

data "oci_containerengine_node_pool_option" "all" {
  node_pool_option_id = "all"
  compartment_id      = var.compartment_ocid
}

locals {
  k8s = trimprefix(var.kubernetes_version, "v")
  gpu_images = [for s in data.oci_containerengine_node_pool_option.all.sources : s.image_id
  if length(regexall("GPU", s.source_name)) > 0 && length(regexall("OKE-${local.k8s}", s.source_name)) > 0]
  cpu_images = [for s in data.oci_containerengine_node_pool_option.all.sources : s.image_id
    if length(regexall("GPU|aarch64", s.source_name)) == 0 && length(regexall("Oracle-Linux-8", s.source_name)) > 0
  && length(regexall("OKE-${local.k8s}", s.source_name)) > 0]
  ad = data.oci_identity_availability_domains.ads.availability_domains[0].name
}

resource "oci_core_vcn" "vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.50.0.0/16"]
  display_name   = "${var.prefix}-vcn"
  dns_label      = "plvvcn"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  enabled        = true
  display_name   = "${var.prefix}-igw"
}

resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  display_name   = "${var.prefix}-public"
  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

resource "oci_core_security_list" "lab" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  display_name   = "${var.prefix}-lab"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  ingress_security_rules {
    source   = "10.50.0.0/16"
    protocol = "all"
  }

  ingress_security_rules {
    source   = var.api_allowed_cidr
    protocol = "6"
    tcp_options {
      min = 6443
      max = 6443
    }
  }
}

resource "oci_core_subnet" "api" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.vcn.id
  cidr_block        = "10.50.0.0/28"
  display_name      = "${var.prefix}-api"
  dns_label         = "api"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.lab.id]
}

resource "oci_core_subnet" "nodes" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.vcn.id
  cidr_block        = "10.50.10.0/24"
  display_name      = "${var.prefix}-nodes"
  dns_label         = "nodes"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.lab.id]
}

resource "oci_core_subnet" "lb" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.vcn.id
  cidr_block        = "10.50.20.0/24"
  display_name      = "${var.prefix}-lb"
  dns_label         = "lb"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.lab.id]
}

resource "oci_containerengine_cluster" "oke" {
  compartment_id     = var.compartment_ocid
  kubernetes_version = var.kubernetes_version
  name               = "${var.prefix}-oke"
  vcn_id             = oci_core_vcn.vcn.id
  type               = "ENHANCED_CLUSTER"

  cluster_pod_network_options {
    cni_type = "FLANNEL_OVERLAY"
  }

  endpoint_config {
    is_public_ip_enabled = true
    subnet_id            = oci_core_subnet.api.id
  }

  options {
    service_lb_subnet_ids = [oci_core_subnet.lb.id]
    kubernetes_network_config {
      pods_cidr     = "10.244.0.0/16"
      services_cidr = "10.96.0.0/16"
    }
  }
}

resource "oci_containerengine_node_pool" "system" {
  cluster_id         = oci_containerengine_cluster.oke.id
  compartment_id     = var.compartment_ocid
  kubernetes_version = var.kubernetes_version
  name               = "system"
  node_shape         = "VM.Standard.E5.Flex"

  node_shape_config {
    ocpus         = 2
    memory_in_gbs = 16
  }

  node_source_details {
    source_type             = "IMAGE"
    image_id                = local.cpu_images[0]
    boot_volume_size_in_gbs = 100
  }

  node_config_details {
    size = 2
    placement_configs {
      availability_domain = local.ad
      subnet_id           = oci_core_subnet.nodes.id
    }
  }
}

resource "oci_containerengine_node_pool" "gpu" {
  cluster_id         = oci_containerengine_cluster.oke.id
  compartment_id     = var.compartment_ocid
  kubernetes_version = var.kubernetes_version
  name               = "gpu"
  node_shape         = var.gpu_shape

  initial_node_labels {
    key   = "plv.io/pool"
    value = "gpu"
  }

  node_source_details {
    source_type             = "IMAGE"
    image_id                = local.gpu_images[0]
    boot_volume_size_in_gbs = 200
  }

  node_config_details {
    size = 1
    placement_configs {
      availability_domain = local.ad
      subnet_id           = oci_core_subnet.nodes.id
    }
  }
}

resource "oci_kms_vault" "vault" {
  compartment_id = var.compartment_ocid
  display_name   = "${var.prefix}-vault"
  vault_type     = "DEFAULT"
}

resource "oci_kms_key" "secrets" {
  compartment_id      = var.compartment_ocid
  display_name        = "${var.prefix}-secrets-key"
  management_endpoint = oci_kms_vault.vault.management_endpoint
  key_shape {
    algorithm = "AES"
    length    = 32
  }
}

# OCI secrets need initial content: a placeholder, replaced by scripts/put-secrets.sh oke (never a real key here).
resource "oci_vault_secret" "ngc" {
  compartment_id = var.compartment_ocid
  vault_id       = oci_kms_vault.vault.id
  key_id         = oci_kms_key.secrets.id
  secret_name    = "plv-ngc-api-key"
  secret_content {
    content_type = "BASE64"
    content      = base64encode("placeholder-set-by-put-secrets")
  }
}

resource "oci_vault_secret" "hf" {
  compartment_id = var.compartment_ocid
  vault_id       = oci_kms_vault.vault.id
  key_id         = oci_kms_key.secrets.id
  secret_name    = "plv-hf-token"
  secret_content {
    content_type = "BASE64"
    content      = base64encode("placeholder-set-by-put-secrets")
  }
}

# OKE workload identity: only the External Secrets controller's service account in this cluster may read secrets.
resource "oci_identity_policy" "eso" {
  compartment_id = var.compartment_ocid
  name           = "${var.prefix}-eso-read-secrets"
  description    = "External Secrets Operator on plv-oke reads plv secrets"
  statements = [
    "Allow any-user to read secret-family in compartment id ${var.compartment_ocid} where all {request.principal.type = 'workload', request.principal.namespace = 'external-secrets', request.principal.service_account = 'external-secrets', request.principal.cluster_id = '${oci_containerengine_cluster.oke.id}'}"
  ]
}
