# Plover Lab 04 - Azure side: AKS (Cilium network policy, OIDC + workload identity), GPU pool owned by the
# NVIDIA GPU Operator (gpu_driver = "None", 🔁 Lab 02), Key Vault for the NGC key, an identity that External
# Secrets Operator uses through workload identity, and ACR for mirrored images.

data "azurerm_client_config" "current" {}

locals {
  suffix = substr(sha1("${var.subscription_id}/${var.resource_group}"), 0, 6)
  tags = {
    project = "plv-multicloud-nim-lab"
    purpose = "nvidia-sa-training"
    data    = "synthetic-only"
  }
  spot = var.gpu_priority == "Spot"
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group
  location = var.location
  tags     = local.tags
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                      = "${var.prefix}-aks"
  location                  = azurerm_resource_group.rg.location
  resource_group_name       = azurerm_resource_group.rg.name
  dns_prefix                = "${var.prefix}-aks"
  kubernetes_version        = var.kubernetes_version
  sku_tier                  = "Free"
  oidc_issuer_enabled       = true
  workload_identity_enabled = true
  tags                      = local.tags

  default_node_pool {
    name                        = "system"
    vm_size                     = var.system_vm_size
    node_count                  = 2
    os_sku                      = "Ubuntu"
    max_pods                    = 110
    temporary_name_for_rotation = "systemtmp"
  }

  identity {
    type = "SystemAssigned"
  }

  # Manual = classic node pools + cluster autoscaler. "Auto" is Node Auto Provisioning (Karpenter on AKS) -
  # compare it with Karpenter on EKS in docs/07-csp-cheatsheet.md.
  node_provisioning_profile {
    mode = "Manual"
  }

  network_profile {
    network_plugin      = "azure"
    network_plugin_mode = "overlay"
    network_data_plane  = "cilium"
    network_policy      = "cilium"
    load_balancer_sku   = "standard"
    pod_cidr            = "10.244.0.0/16"
    service_cidr        = "10.0.0.0/16"
    dns_service_ip      = "10.0.0.10"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.gpu_vm_size
  os_sku                = "Ubuntu"
  os_disk_size_gb       = 256
  max_pods              = 60
  gpu_driver            = "None" # the GPU Operator installs driver + container toolkit (never mix the two modes)
  priority              = var.gpu_priority
  eviction_policy       = local.spot ? "Delete" : null
  spot_max_price        = local.spot ? -1 : null
  auto_scaling_enabled  = true
  min_count             = 0
  max_count             = var.gpu_max_nodes
  node_count            = 1
  node_labels = merge(
    { "plv.io/pool" = "gpu", "nvidia.com/mig.config" = "all-disabled" },
    local.spot ? { "kubernetes.azure.com/scalesetpriority" = "spot" } : {}
  )
  node_taints = concat(
    ["nvidia.com/gpu=present:NoSchedule"],
    local.spot ? ["kubernetes.azure.com/scalesetpriority=spot:NoSchedule"] : []
  )
  tags = local.tags

  lifecycle {
    ignore_changes = [node_count]
  }
}

resource "azurerm_key_vault" "kv" {
  name                       = "${var.prefix}kv${local.suffix}"
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  rbac_authorization_enabled = true
  soft_delete_retention_days = 7
  purge_protection_enabled   = false
  tags                       = local.tags
}

# You upload secret values (scripts/put-secrets.sh); Terraform never sees them, so they never land in state.
resource "azurerm_role_assignment" "me_kv_officer" {
  scope                = azurerm_key_vault.kv.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

resource "azurerm_user_assigned_identity" "eso" {
  name                = "${var.prefix}-eso"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = local.tags
}

resource "azurerm_federated_identity_credential" "eso" {
  name                      = "external-secrets"
  user_assigned_identity_id = azurerm_user_assigned_identity.eso.id
  audience                  = ["api://AzureADTokenExchange"]
  issuer                    = azurerm_kubernetes_cluster.aks.oidc_issuer_url
  subject                   = "system:serviceaccount:external-secrets:external-secrets"
}

resource "azurerm_role_assignment" "eso_kv_reader" {
  scope                = azurerm_key_vault.kv.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.eso.principal_id
}

resource "azurerm_container_registry" "acr" {
  name                = "${var.prefix}acr${local.suffix}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "Standard"
  admin_enabled       = false
  tags                = local.tags
}

resource "azurerm_role_assignment" "kubelet_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
}
