# Main Terraform configuration for MLOps Fraud Detection Pipeline
# Deploys complete infrastructure to AWS using ECS Fargate, RDS, S3, ALB

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Backend configuration for remote state
  backend "s3" {
    bucket         = "mlops-fraud-detection-terraform-state"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "MLOps-Fraud-Detection"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "MLOps-Team"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local variables
locals {
  name_prefix = "mlops-fraud-${var.environment}"
  common_tags = {
    Project     = "MLOps-Fraud-Detection"
    Environment = var.environment
  }
}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"

  name_prefix         = local.name_prefix
  vpc_cidr            = var.vpc_cidr
  availability_zones  = slice(data.aws_availability_zones.available.names, 0, 2)
  public_subnet_cidrs = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs

  tags = local.common_tags
}

# Security Groups
module "security_groups" {
  source = "./modules/security"

  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id

  tags = local.common_tags
}

# S3 Buckets for Data and Artifacts
module "s3" {
  source = "./modules/s3"

  name_prefix = local.name_prefix
  environment = var.environment

  tags = local.common_tags
}

# RDS PostgreSQL for Airflow and MLflow
module "rds" {
  source = "./modules/rds"

  name_prefix           = local.name_prefix
  vpc_id                = module.vpc.vpc_id
  private_subnet_ids    = module.vpc.private_subnet_ids
  security_group_ids    = [module.security_groups.rds_sg_id]

  db_name               = var.db_name
  db_username           = var.db_username
  db_password           = var.db_password
  instance_class        = var.rds_instance_class
  allocated_storage     = var.rds_allocated_storage
  multi_az              = var.rds_multi_az
  backup_retention_days = var.rds_backup_retention_days

  tags = local.common_tags
}

# ECR Repositories for Docker Images
module "ecr" {
  source = "./modules/ecr"

  name_prefix = local.name_prefix
  repositories = [
    "fraud-detector-api",
    "mlflow-server",
    "airflow-webserver",
    "airflow-scheduler",
  ]

  tags = local.common_tags
}

# ECS Cluster
module "ecs" {
  source = "./modules/ecs"

  name_prefix = local.name_prefix
  environment = var.environment

  tags = local.common_tags
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"

  name_prefix        = local.name_prefix
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  security_group_ids = [module.security_groups.alb_sg_id]

  certificate_arn = var.acm_certificate_arn  # For HTTPS

  tags = local.common_tags
}

# FastAPI Service
module "fastapi_service" {
  source = "./modules/ecs-service"

  name_prefix         = "${local.name_prefix}-api"
  cluster_id          = module.ecs.cluster_id
  cluster_name        = module.ecs.cluster_name
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  security_group_ids  = [module.security_groups.ecs_sg_id]

  # Task Definition
  container_name      = "fraud-detector-api"
  container_image     = "${module.ecr.repository_urls["fraud-detector-api"]}:latest"
  container_port      = 8000
  cpu                 = 1024  # 1 vCPU
  memory              = 2048  # 2 GB
  desired_count       = var.api_desired_count

  # Environment Variables
  environment_variables = {
    ENVIRONMENT              = var.environment
    MLFLOW_TRACKING_URI      = "http://${module.mlflow_service.service_endpoint}:5000"
    AWS_REGION               = var.aws_region
    S3_BUCKET_NAME           = module.s3.data_bucket_name
    MODEL_STAGE              = "Production"
    LOG_LEVEL                = "INFO"
  }

  # Secrets from AWS Secrets Manager
  secrets = {
    DB_PASSWORD = "${module.rds.db_secret_arn}:password::"
  }

  # Load Balancer
  target_group_arn = module.alb.target_group_arns["api"]
  health_check_path = "/health"

  # Auto Scaling
  autoscaling_enabled = true
  autoscaling_min_capacity = var.api_min_capacity
  autoscaling_max_capacity = var.api_max_capacity
  autoscaling_cpu_threshold = 70
  autoscaling_memory_threshold = 80

  tags = local.common_tags
}

# MLflow Service
module "mlflow_service" {
  source = "./modules/ecs-service"

  name_prefix         = "${local.name_prefix}-mlflow"
  cluster_id          = module.ecs.cluster_id
  cluster_name        = module.ecs.cluster_name
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  security_group_ids  = [module.security_groups.ecs_sg_id]

  # Task Definition
  container_name      = "mlflow-server"
  container_image     = "${module.ecr.repository_urls["mlflow-server"]}:latest"
  container_port      = 5000
  cpu                 = 512   # 0.5 vCPU
  memory              = 1024  # 1 GB
  desired_count       = 1     # Single instance (not HA)

  # Environment Variables
  environment_variables = {
    BACKEND_STORE_URI = "postgresql://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}/${var.db_name}"
    ARTIFACT_ROOT     = "s3://${module.s3.artifact_bucket_name}/mlflow"
    AWS_REGION        = var.aws_region
  }

  # Load Balancer
  target_group_arn = module.alb.target_group_arns["mlflow"]
  health_check_path = "/health"

  # No autoscaling for MLflow (single instance)
  autoscaling_enabled = false

  tags = local.common_tags
}

# Airflow Webserver Service
module "airflow_webserver_service" {
  source = "./modules/ecs-service"

  name_prefix         = "${local.name_prefix}-airflow-web"
  cluster_id          = module.ecs.cluster_id
  cluster_name        = module.ecs.cluster_name
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  security_group_ids  = [module.security_groups.ecs_sg_id]

  # Task Definition
  container_name      = "airflow-webserver"
  container_image     = "${module.ecr.repository_urls["airflow-webserver"]}:latest"
  container_port      = 8080
  cpu                 = 1024  # 1 vCPU
  memory              = 2048  # 2 GB
  desired_count       = 1

  # Environment Variables
  environment_variables = {
    AIRFLOW__CORE__SQL_ALCHEMY_CONN = "postgresql://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}/${var.db_name}"
    AIRFLOW__CORE__EXECUTOR         = "LocalExecutor"
    AIRFLOW__CORE__LOAD_EXAMPLES    = "False"
    AWS_REGION                      = var.aws_region
  }

  # Load Balancer
  target_group_arn = module.alb.target_group_arns["airflow"]
  health_check_path = "/health"

  autoscaling_enabled = false

  tags = local.common_tags
}

# Airflow Scheduler Service
module "airflow_scheduler_service" {
  source = "./modules/ecs-service"

  name_prefix         = "${local.name_prefix}-airflow-scheduler"
  cluster_id          = module.ecs.cluster_id
  cluster_name        = module.ecs.cluster_name
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  security_group_ids  = [module.security_groups.ecs_sg_id]

  # Task Definition
  container_name      = "airflow-scheduler"
  container_image     = "${module.ecr.repository_urls["airflow-scheduler"]}:latest"
  container_port      = 8793  # Scheduler doesn't need LB
  cpu                 = 1024
  memory              = 2048
  desired_count       = 1

  # Environment Variables
  environment_variables = {
    AIRFLOW__CORE__SQL_ALCHEMY_CONN = "postgresql://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}/${var.db_name}"
    AIRFLOW__CORE__EXECUTOR         = "LocalExecutor"
    AWS_REGION                      = var.aws_region
  }

  # No load balancer for scheduler
  target_group_arn = null

  autoscaling_enabled = false

  tags = local.common_tags
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "ecs_logs" {
  for_each = toset([
    "fraud-detector-api",
    "mlflow-server",
    "airflow-webserver",
    "airflow-scheduler",
  ])

  name              = "/ecs/${local.name_prefix}/${each.value}"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# CloudWatch Alarms
module "cloudwatch_alarms" {
  source = "./modules/cloudwatch"

  name_prefix = local.name_prefix

  # API Service Alarms
  api_service_name = module.fastapi_service.service_name
  cluster_name     = module.ecs.cluster_name

  # Thresholds
  high_cpu_threshold    = 80
  high_memory_threshold = 85
  error_rate_threshold  = 5
  latency_threshold     = 500  # ms

  # SNS Topic for Alerts
  alarm_sns_topic_arn = module.sns.topic_arn

  tags = local.common_tags
}

# SNS for Alerts
module "sns" {
  source = "./modules/sns"

  name_prefix = local.name_prefix
  email_addresses = var.alert_email_addresses

  tags = local.common_tags
}

# IAM Roles and Policies
module "iam" {
  source = "./modules/iam"

  name_prefix = local.name_prefix

  # S3 Buckets
  data_bucket_arn     = module.s3.data_bucket_arn
  artifact_bucket_arn = module.s3.artifact_bucket_arn

  # ECR Repositories
  ecr_repository_arns = values(module.ecr.repository_arns)

  tags = local.common_tags
}

# Route53 DNS (Optional)
# Uncomment if you have a domain
# module "route53" {
#   source = "./modules/route53"
#
#   domain_name = var.domain_name
#   alb_dns_name = module.alb.dns_name
#   alb_zone_id  = module.alb.zone_id
#
#   tags = local.common_tags
# }

# Outputs
output "api_endpoint" {
  description = "FastAPI endpoint URL"
  value       = "https://${module.alb.dns_name}"
}

output "mlflow_endpoint" {
  description = "MLflow UI endpoint URL"
  value       = "http://${module.alb.dns_name}:5000"
}

output "airflow_endpoint" {
  description = "Airflow UI endpoint URL"
  value       = "http://${module.alb.dns_name}:8080"
}

output "ecr_repositories" {
  description = "ECR repository URLs"
  value       = module.ecr.repository_urls
}

output "rds_endpoint" {
  description = "RDS database endpoint"
  value       = module.rds.db_endpoint
  sensitive   = true
}

output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    data     = module.s3.data_bucket_name
    artifacts = module.s3.artifact_bucket_name
  }
}
