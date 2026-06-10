terraform {
  required_version = ">= 1.5.0"
}

# Minimal placeholder scaffold for production evolution.
# Extend with your cloud provider modules for:
# - object storage
# - managed Postgres
# - container registry
# - Kubernetes cluster
# - secrets manager

output "project_name" {
  value = "docintel-ai-platform"
}
