output "cluster_name" { value = google_container_cluster.gke.name }
output "zone" { value = var.zone }
output "project" { value = var.project }
output "ar_repo" { value = "${var.region}-docker.pkg.dev/${var.project}/${google_artifact_registry_repository.plv.repository_id}" }
output "gpu_machine_type" { value = local.gpu.machine }
output "gpu_accelerator" { value = local.gpu.accelerator }
