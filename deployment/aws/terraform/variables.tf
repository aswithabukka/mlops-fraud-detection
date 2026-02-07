# Terraform Variables for MLOps Fraud Detection Pipeline

# General
variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "mlops-fraud-detection"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24"]
}

# RDS Configuration
variable "db_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "mlflow"
}

variable "db_username" {
  description = "Master username for RDS database"
  type        = string
  default     = "mlflow"
  sensitive   = true
}

variable "db_password" {
  description = "Master password for RDS database"
  type        = string
  sensitive   = true
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.small"

  validation {
    condition     = can(regex("^db\\.", var.rds_instance_class))
    error_message = "RDS instance class must start with 'db.'."
  }
}

variable "rds_allocated_storage" {
  description = "Allocated storage for RDS in GB"
  type        = number
  default     = 20

  validation {
    condition     = var.rds_allocated_storage >= 20 && var.rds_allocated_storage <= 1000
    error_message = "RDS allocated storage must be between 20 and 1000 GB."
  }
}

variable "rds_multi_az" {
  description = "Enable Multi-AZ for RDS (high availability)"
  type        = bool
  default     = false  # Set to true for production
}

variable "rds_backup_retention_days" {
  description = "Number of days to retain RDS backups"
  type        = number
  default     = 7

  validation {
    condition     = var.rds_backup_retention_days >= 1 && var.rds_backup_retention_days <= 35
    error_message = "Backup retention must be between 1 and 35 days."
  }
}

# ECS Configuration
variable "api_desired_count" {
  description = "Desired number of FastAPI service tasks"
  type        = number
  default     = 2
}

variable "api_min_capacity" {
  description = "Minimum number of FastAPI tasks (autoscaling)"
  type        = number
  default     = 2
}

variable "api_max_capacity" {
  description = "Maximum number of FastAPI tasks (autoscaling)"
  type        = number
  default     = 10
}

variable "api_cpu" {
  description = "CPU units for FastAPI task (1024 = 1 vCPU)"
  type        = number
  default     = 1024
}

variable "api_memory" {
  description = "Memory for FastAPI task in MB"
  type        = number
  default     = 2048
}

# S3 Configuration
variable "s3_versioning_enabled" {
  description = "Enable versioning for S3 buckets"
  type        = bool
  default     = true
}

variable "s3_lifecycle_days" {
  description = "Days after which to transition S3 objects to cheaper storage"
  type        = number
  default     = 90
}

# CloudWatch Configuration
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

# SSL/TLS Configuration
variable "acm_certificate_arn" {
  description = "ARN of ACM certificate for HTTPS (optional)"
  type        = string
  default     = ""
}

# Alerting Configuration
variable "alert_email_addresses" {
  description = "Email addresses to receive CloudWatch alarms"
  type        = list(string)
  default     = []
}

variable "enable_slack_alerts" {
  description = "Enable Slack alerts for critical events"
  type        = bool
  default     = false
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  default     = ""
  sensitive   = true
}

# Monitoring Configuration
variable "enable_enhanced_monitoring" {
  description = "Enable enhanced RDS monitoring (additional cost)"
  type        = bool
  default     = false
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval in seconds (0, 1, 5, 10, 15, 30, 60)"
  type        = number
  default     = 60

  validation {
    condition     = contains([0, 1, 5, 10, 15, 30, 60], var.monitoring_interval)
    error_message = "Monitoring interval must be 0, 1, 5, 10, 15, 30, or 60 seconds."
  }
}

# Cost Optimization
variable "use_spot_instances" {
  description = "Use Fargate Spot for cost savings (70% cheaper, less reliable)"
  type        = bool
  default     = false
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for RDS"
  type        = bool
  default     = true  # Prevent accidental deletion in production
}

# Tagging
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Domain Configuration (Optional)
variable "domain_name" {
  description = "Domain name for the application (optional)"
  type        = string
  default     = ""
}

variable "create_route53_records" {
  description = "Create Route53 DNS records (requires domain_name)"
  type        = bool
  default     = false
}
