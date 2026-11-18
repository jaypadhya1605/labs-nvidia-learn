output "cluster_name" { value = aws_eks_cluster.eks.name }
output "region" { value = var.region }
output "ecr_repo" { value = aws_ecr_repository.plv.repository_url }
