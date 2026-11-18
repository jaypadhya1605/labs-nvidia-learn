output "cluster_id" { value = oci_containerengine_cluster.oke.id }
output "region" { value = var.region }
output "vault_ocid" { value = oci_kms_vault.vault.id }
output "secret_ocid_ngc" { value = oci_vault_secret.ngc.id }
output "secret_ocid_hf" { value = oci_vault_secret.hf.id }
