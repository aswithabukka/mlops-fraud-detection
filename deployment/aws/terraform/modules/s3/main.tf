# S3 Module - Creates buckets for data and ML artifacts

# Data Bucket - Stores raw and processed transaction data
resource "aws_s3_bucket" "data" {
  bucket = "${var.name_prefix}-data-${var.environment}"

  tags = merge(
    var.tags,
    {
      Name    = "${var.name_prefix}-data-bucket"
      Purpose = "Transaction data storage"
    }
  )
}

# Artifact Bucket - Stores MLflow models and artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.name_prefix}-artifacts-${var.environment}"

  tags = merge(
    var.tags,
    {
      Name    = "${var.name_prefix}-artifacts-bucket"
      Purpose = "MLflow artifact storage"
    }
  )
}

# Enable versioning for data bucket
resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable versioning for artifacts bucket
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption for data bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Server-side encryption for artifacts bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access for data bucket
resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Block public access for artifacts bucket
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy for data bucket - transition to cheaper storage
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "transition-old-data"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "STANDARD_IA"  # Infrequent Access
    }

    transition {
      days          = 180
      storage_class = "GLACIER"  # Archive
    }

    expiration {
      days = 365  # Delete after 1 year
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# Lifecycle policy for artifacts bucket
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "transition-old-artifacts"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    # Don't delete artifacts automatically
    noncurrent_version_expiration {
      noncurrent_days = 180  # Keep old model versions for 6 months
    }
  }
}

# Outputs
output "data_bucket_name" {
  description = "Name of the data S3 bucket"
  value       = aws_s3_bucket.data.id
}

output "data_bucket_arn" {
  description = "ARN of the data S3 bucket"
  value       = aws_s3_bucket.data.arn
}

output "artifact_bucket_name" {
  description = "Name of the artifacts S3 bucket"
  value       = aws_s3_bucket.artifacts.id
}

output "artifact_bucket_arn" {
  description = "ARN of the artifacts S3 bucket"
  value       = aws_s3_bucket.artifacts.arn
}
