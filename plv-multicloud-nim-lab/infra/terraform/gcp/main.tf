# Plover Lab 04 - Google Cloud side: VPC-native zonal GKE Standard with Dataplane V2 (enforces NetworkPolicy,
# like Cilium on AKS), Workload Identity Federation, a GPU pool prepared for the NVIDIA GPU Operator
# ("Google driver installer" path: GKE driver install disabled + GKE device plugin disabled by node label),
# Secret Manager for the NGC key, Artifact Registry for mirrored images (image streaming in M7).

locals {
  gpu_profiles = {
    rtxpro6000 = { machine = "g4-standard-48", accelerator = "nvidia-rtx-pro-6000", disk = "hyperdisk-balanced" }
    a100-40    = { machine = "a2-highgpu-1g", accelerator = "nvidia-tesla-a100", disk = "pd-balanced" }
    l4         = { machine = "g2-standard-8", accelerator = "nvidia-l4", disk = "pd-balanced" }
  }
  gpu           = local.gpu_profiles[var.gpu_profile]
  eso_principal = "principal://iam.googleapis.com/projects/${data.google_project.this.number}/locations/global/workloadIdentityPools/${var.project}.svc.id.goog/subject/ns/external-secrets/sa/external-secrets"
  labels = {
    project = "plv-multicloud-nim-lab"
    purpose = "nvidia-sa-training"
  }
}

data "google_project" "this" {
  project_id = var.project
}

resource "google_project_service" "apis" {
  for_each           = toset(["compute.googleapis.com", "container.googleapis.com", "secretmanager.googleapis.com", "artifactregistry.googleapis.com"])
  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "vpc" {
  name                    = "${var.prefix}-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "nodes" {
  name          = "${var.prefix}-nodes"
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = "10.10.0.0/20"

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.20.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.30.0.0/20"
  }
}

resource "google_container_cluster" "gke" {
  name                     = "${var.prefix}-gke"
  location                 = var.zone
  network                  = google_compute_network.vpc.id
  subnetwork               = google_compute_subnetwork.nodes.id
  networking_mode          = "VPC_NATIVE"
  datapath_provider        = "ADVANCED_DATAPATH"
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false
  resource_labels          = local.labels

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  release_channel {
    channel = "REGULAR" # G4 needs GKE >= 1.34.0-gke.1662000; REGULAR is past that in Oct 2026 - verify with `gcloud container get-server-config`
  }

  workload_identity_config {
    workload_pool = "${var.project}.svc.id.goog"
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = false # we run kube-prometheus-stack in-cluster on every cloud (🔁 Lab 02) for identical queries
    }
  }
}

resource "google_container_node_pool" "system" {
  name       = "system"
  cluster    = google_container_cluster.gke.id
  location   = var.zone
  node_count = 2

  node_config {
    machine_type    = "e2-standard-4"
    disk_size_gb    = 100
    resource_labels = local.labels
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "gpu" {
  name               = "gpu"
  cluster            = google_container_cluster.gke.id
  location           = var.zone
  initial_node_count = 1

  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = local.gpu.machine
    disk_type       = local.gpu.disk
    disk_size_gb    = 200
    image_type      = "COS_CONTAINERD"
    spot            = var.gpu_spot
    resource_labels = local.labels
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = {
      "plv.io/pool"                             = "gpu"
      "gke-no-default-nvidia-gpu-device-plugin" = "true" # the GPU Operator's device plugin takes over
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    guest_accelerator {
      type  = local.gpu.accelerator
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "INSTALLATION_DISABLED" # Google's driver-installer DaemonSet installs it (GitOps app gke-driver-installer)
      }
    }

    gcfs_config {
      enabled = true # image streaming for images in Artifact Registry (M7 cold-start experiment)
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  lifecycle {
    ignore_changes = [initial_node_count]
  }
}

# Secret containers only; you add versions with scripts/put-secrets.sh gke.
resource "google_secret_manager_secret" "ngc" {
  secret_id = "plv-ngc-api-key"
  labels    = local.labels
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "hf" {
  secret_id = "plv-hf-token"
  labels    = local.labels
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

# GKE Workload Identity Federation: grant the External Secrets controller's Kubernetes service account directly
# (no Google service account, no keys).
resource "google_secret_manager_secret_iam_member" "eso_ngc" {
  secret_id = google_secret_manager_secret.ngc.id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.eso_principal
}

resource "google_secret_manager_secret_iam_member" "eso_hf" {
  secret_id = google_secret_manager_secret.hf.id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.eso_principal
}

resource "google_artifact_registry_repository" "plv" {
  location      = var.region
  repository_id = "${var.prefix}-images"
  format        = "DOCKER"
  labels        = local.labels
  depends_on    = [google_project_service.apis]
}
