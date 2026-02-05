terraform {
  required_providers {
    runpod = {
      source = "decentralized-infrastructure/runpod"
      version = "1.0.1"
    }
  }
}

variable "runpod_api_key" {
  type = string
  sensitive = true
  description = "API KEY for RUNPOD"
}

variable "jupyter_password" {
    type = string
    sensitive = true
    description = "The password for jupyterlab"
}

variable "public_key" {
    type = string
    sensitive = true
    description = "Key for SSH communication"
}

provider "runpod" {
  api_key = var.runpod_api_key
}

resource "runpod_pod" "alexandria_pod" {
  name              = "alexandria-pod-v1"
  image_name        = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
  template_id       = "runpod-torch-v280"
  gpu_type_ids      = ["NVIDIA GeForce RTX 4090"]
  data_center_ids   = ["US-CA-2", "US-TX-3"]

  gpu_count            = 1
  cloud_type           = "SECURE"
  support_public_ip    = true

  volume_in_gb         = 30
  container_disk_in_gb = 30

  env = {
    JUPYTER_PASSWORD = var.jupyter_password,
    PUBLIC_KEY = var.public_key
  }
}