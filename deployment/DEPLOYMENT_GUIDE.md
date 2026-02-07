# MLOps Fraud Detection - Deployment Guide

Complete guide to deploying the fraud detection pipeline to AWS using Terraform and GitHub Actions CI/CD.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [AWS Account Setup](#aws-account-setup)
4. [Terraform Deployment](#terraform-deployment)
5. [Docker Image Build & Push](#docker-image-build--push)
6. [GitHub Actions CI/CD Setup](#github-actions-cicd-setup)
7. [Verification](#verification)
8. [Monitoring & Operations](#monitoring--operations)
9. [Troubleshooting](#troubleshooting)
10. [Cost Optimization](#cost-optimization)

---

## Prerequisites

### Required Tools

```bash
# Verify installations
terraform --version  # >= 1.7.0
aws --version        # >= 2.0
docker --version     # >= 24.0
git --version        # >= 2.0

# Install if missing:
# - Terraform: https://www.terraform.io/downloads
# - AWS CLI: https://aws.amazon.com/cli/
# - Docker: https://docs.docker.com/get-docker/
```

### AWS Requirements

- AWS Account with admin access (or appropriate IAM permissions)
- AWS CLI configured with credentials
- Domain name (optional, for HTTPS)
- ACM certificate (optional, for SSL/TLS)

### Knowledge Requirements

- Basic AWS services (VPC, ECS, RDS, S3, ALB)
- Docker containerization
- Terraform basics
- GitHub Actions (for CI/CD)

---

## Local Development Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd MLOps
```

### 2. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test (optional)
pre-commit run --all-files
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your values
vim .env
```

### 4. Test Locally

```bash
# Start all services
make docker-up

# Run tests
make test

# Verify services
make health-check
```

---

## AWS Account Setup

### 1. Create IAM User for Deployment

```bash
# Create user with programmatic access
aws iam create-user --user-name mlops-deployer

# Attach necessary policies
aws iam attach-user-policy \
  --user-name mlops-deployer \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

# Create access keys
aws iam create-access-key --user-name mlops-deployer

# Save access key ID and secret access key
```

**Principle of Least Privilege**: In production, create custom policy with only required permissions instead of AdministratorAccess.

### 2. Configure AWS CLI

```bash
# Configure default profile
aws configure

# Or create named profile
aws configure --profile mlops

# Verify
aws sts get-caller-identity
```

### 3. Create S3 Bucket for Terraform State

```bash
# Create state bucket (one-time)
aws s3 mb s3://mlops-fraud-detection-terraform-state \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket mlops-fraud-detection-terraform-state \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket mlops-fraud-detection-terraform-state \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region us-east-1
```

### 4. Create ACM Certificate (Optional - for HTTPS)

```bash
# Request certificate for your domain
aws acm request-certificate \
  --domain-name fraud-detector.example.com \
  --subject-alternative-names "*.fraud-detector.example.com" \
  --validation-method DNS \
  --region us-east-1

# Follow DNS validation instructions in AWS Console
# Save certificate ARN for later
```

---

## Terraform Deployment

### 1. Navigate to Terraform Directory

```bash
cd deployment/aws/terraform
```

### 2. Configure Variables

```bash
# Copy example variables
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
vim terraform.tfvars
```

**Required variables**:
```hcl
aws_region  = "us-east-1"
environment = "production"

db_username = "mlflow"
db_password = "YOUR_STRONG_PASSWORD_HERE"  # Use AWS Secrets Manager in prod

alert_email_addresses = ["your-email@example.com"]

# Optional: For HTTPS
acm_certificate_arn = "arn:aws:acm:us-east-1:..."
```

### 3. Initialize Terraform

```bash
# Initialize backend and download providers
terraform init

# Verify configuration
terraform validate
```

### 4. Plan Deployment

```bash
# Preview infrastructure changes
terraform plan

# Save plan to file (optional)
terraform plan -out=tfplan
```

**Review carefully**:
- Resources to be created
- Estimated costs
- Security groups and IAM policies

### 5. Apply Infrastructure

```bash
# Apply changes (will prompt for confirmation)
terraform apply

# Or apply saved plan
terraform apply tfplan

# This will take 10-15 minutes
```

**What gets created**:
- VPC with public/private subnets (2 AZs)
- Internet Gateway, NAT Gateways
- RDS PostgreSQL database
- ECS Fargate cluster
- Application Load Balancer
- ECR repositories (4)
- S3 buckets (2)
- CloudWatch log groups
- IAM roles and policies
- Security groups
- SNS topic for alerts

### 6. Save Outputs

```bash
# Display outputs
terraform output

# Save important values
export API_ENDPOINT=$(terraform output -raw api_endpoint)
export ECR_API_REPO=$(terraform output -json ecr_repositories | jq -r '."fraud-detector-api"')
export ECR_MLFLOW_REPO=$(terraform output -json ecr_repositories | jq -r '."mlflow-server"')

echo "API Endpoint: $API_ENDPOINT"
echo "ECR API Repo: $ECR_API_REPO"
echo "ECR MLflow Repo: $ECR_MLFLOW_REPO"
```

---

## Docker Image Build & Push

### 1. Login to ECR

```bash
# Get ECR login command
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com
```

### 2. Build Docker Images

```bash
# Return to project root
cd /Users/aswithabukka/CascadeProjects/MLOps

# Build FastAPI image
docker build -t fraud-detector-api:latest -f docker/fastapi.Dockerfile .

# Build MLflow image
docker build -t mlflow-server:latest -f mlflow/Dockerfile ./mlflow

# Build Airflow images (if needed)
docker build -t airflow-webserver:latest -f airflow/Dockerfile ./airflow
```

### 3. Tag Images

```bash
# Tag for ECR
docker tag fraud-detector-api:latest $ECR_API_REPO:latest
docker tag mlflow-server:latest $ECR_MLFLOW_REPO:latest
```

### 4. Push to ECR

```bash
# Push images
docker push $ECR_API_REPO:latest
docker push $ECR_MLFLOW_REPO:latest

# Verify
aws ecr list-images --repository-name fraud-detector-api
```

### 5. Update ECS Services

```bash
# Force new deployment with latest images
aws ecs update-service \
  --cluster mlops-fraud-production \
  --service fraud-detector-api-service \
  --force-new-deployment

# Monitor deployment
aws ecs wait services-stable \
  --cluster mlops-fraud-production \
  --services fraud-detector-api-service
```

---

## GitHub Actions CI/CD Setup

### 1. Add GitHub Secrets

Navigate to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

```
AWS_ACCESS_KEY_ID          = <your-access-key-id>
AWS_SECRET_ACCESS_KEY      = <your-secret-access-key>
SLACK_WEBHOOK_URL          = <optional-slack-webhook>
```

### 2. Configure Branch Protection

Settings → Branches → Add branch protection rule for `main`:

- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
  - Select: `lint`, `test-unit`, `test-integration`, `docker-build`
- ✅ Require branches to be up to date before merging

### 3. Enable Actions

- Ensure GitHub Actions are enabled in repository settings
- Review workflows in `.github/workflows/`

### 4. Test CI Pipeline

```bash
# Create feature branch
git checkout -b feature/test-ci

# Make a small change
echo "# Test CI" >> README.md

# Commit and push
git add README.md
git commit -m "test: Verify CI pipeline"
git push origin feature/test-ci

# Create pull request
# Watch CI pipeline run in GitHub Actions tab
```

**CI Pipeline Steps**:
1. ✅ Code Quality (black, flake8, isort, mypy, bandit)
2. ✅ Unit Tests (with coverage upload)
3. ✅ Integration Tests (with PostgreSQL service)
4. ✅ Docker Build (test images build successfully)
5. ✅ Security Scan (Trivy vulnerability scan)

### 5. Test CD Pipeline

```bash
# Merge PR to main
# CD pipeline automatically triggers

# Watch deployment:
# 1. Build & push Docker images to ECR
# 2. Deploy to staging
# 3. Run smoke tests
# 4. Wait for manual approval
# 5. Deploy to production
# 6. Tag release
```

**Manual Approval**:
- Navigate to Actions tab
- Click on CD workflow run
- Click "Review deployments"
- Approve production deployment

---

## Verification

### 1. Check Infrastructure

```bash
# Verify ECS services are running
aws ecs list-services --cluster mlops-fraud-production

# Check service tasks
aws ecs list-tasks --cluster mlops-fraud-production \
  --service-name fraud-detector-api-service

# Describe task to get public IP (if needed)
TASK_ARN=$(aws ecs list-tasks --cluster mlops-fraud-production \
  --service-name fraud-detector-api-service \
  --query 'taskArns[0]' --output text)

aws ecs describe-tasks --cluster mlops-fraud-production \
  --tasks $TASK_ARN
```

### 2. Test API Endpoints

```bash
# Get ALB DNS name
ALB_DNS=$(terraform output -raw api_endpoint)

# Test health endpoint
curl $ALB_DNS/health

# Test prediction endpoint
curl -X POST "$ALB_DNS/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 250.0,
    "merchant_category": "online_retail",
    "merchant_country": "USA",
    "hour_of_day": 14,
    "day_of_week": 3,
    "is_online": true,
    "is_weekend": false,
    "transactions_last_24h": 2,
    "total_amount_last_24h": 500.0,
    "transactions_last_1h": 0,
    "distance_from_home": 5.0
  }'
```

### 3. Check MLflow UI

```bash
# Access MLflow (internal only, use bastion or VPN)
# Or configure ALB listener rule

open http://$ALB_DNS:5000
```

### 4. Verify CloudWatch Logs

```bash
# View API logs
aws logs tail /ecs/mlops-fraud-production/fraud-detector-api --follow

# View specific log stream
aws logs get-log-events \
  --log-group-name /ecs/mlops-fraud-production/fraud-detector-api \
  --log-stream-name <stream-name>
```

### 5. Check CloudWatch Alarms

```bash
# List alarms
aws cloudwatch describe-alarms \
  --alarm-name-prefix mlops-fraud

# Check alarm state
aws cloudwatch describe-alarms \
  --state-value ALARM
```

---

## Monitoring & Operations

### CloudWatch Dashboards

Navigate to AWS Console → CloudWatch → Dashboards

**Created dashboards**:
- API Performance (latency, throughput, errors)
- ECS Cluster (CPU, memory, task count)
- RDS Database (connections, CPU, storage)

### Viewing Logs

```bash
# Follow logs in real-time
aws logs tail /ecs/mlops-fraud-production/fraud-detector-api --follow

# Search logs
aws logs filter-log-events \
  --log-group-name /ecs/mlops-fraud-production/fraud-detector-api \
  --filter-pattern "ERROR"
```

### Scaling Services

```bash
# Manual scaling
aws ecs update-service \
  --cluster mlops-fraud-production \
  --service fraud-detector-api-service \
  --desired-count 5

# Auto-scaling is configured via Terraform (CPU > 70% scales out)
```

### Database Backups

```bash
# List DB snapshots
aws rds describe-db-snapshots \
  --db-instance-identifier mlops-fraud-production-rds

# Create manual snapshot
aws rds create-db-snapshot \
  --db-instance-identifier mlops-fraud-production-rds \
  --db-snapshot-identifier manual-snapshot-$(date +%Y%m%d-%H%M%S)
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check service events
aws ecs describe-services \
  --cluster mlops-fraud-production \
  --services fraud-detector-api-service \
  --query 'services[0].events[:5]'

# Check task stopped reason
aws ecs describe-tasks \
  --cluster mlops-fraud-production \
  --tasks <task-arn> \
  --query 'tasks[0].stoppedReason'

# Check CloudWatch logs for errors
aws logs tail /ecs/mlops-fraud-production/fraud-detector-api --since 1h
```

### High Latency

```bash
# Check ECS service metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=fraud-detector-api-service \
              Name=ClusterName,Value=mlops-fraud-production \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average

# Check ALB target health
aws elbv2 describe-target-health \
  --target-group-arn <target-group-arn>
```

### Database Connection Issues

```bash
# Check RDS instance status
aws rds describe-db-instances \
  --db-instance-identifier mlops-fraud-production-rds \
  --query 'DBInstances[0].DBInstanceStatus'

# Check security group rules
aws ec2 describe-security-groups \
  --group-ids <rds-security-group-id>

# Test connectivity from ECS task
aws ecs execute-command \
  --cluster mlops-fraud-production \
  --task <task-arn> \
  --container fraud-detector-api \
  --interactive \
  --command "nc -zv <rds-endpoint> 5432"
```

### Terraform State Issues

```bash
# If state is locked
aws dynamodb delete-item \
  --table-name terraform-state-lock \
  --key '{"LockID": {"S": "mlops-fraud-detection-terraform-state/terraform.tfstate"}}'

# Pull latest state
terraform refresh

# Resolve state conflicts
terraform state list
terraform state show <resource>
```

---

## Cost Optimization

### Current Estimated Costs

**Production (default config)**:
- RDS db.t3.small: ~$30/month
- ECS Fargate (2 tasks, 1 vCPU, 2GB): ~$60/month
- ALB: ~$20/month
- NAT Gateways (2): ~$64/month
- S3: ~$5/month
- CloudWatch: ~$5/month
- **Total: ~$185/month**

### Optimization Strategies

#### 1. Use Fargate Spot (70% savings)

```hcl
# In terraform.tfvars
use_spot_instances = true
```

Savings: ~$42/month

#### 2. Single NAT Gateway

```hcl
# Edit modules/vpc/main.tf
# Change NAT gateway count from 2 to 1
```

Savings: ~$32/month

#### 3. RDS Reserved Instances

```bash
# Purchase 1-year reservation
aws rds purchase-reserved-db-instances-offering \
  --reserved-db-instances-offering-id <offering-id> \
  --db-instance-count 1
```

Savings: ~40% (~$12/month)

#### 4. S3 Intelligent Tiering

Already configured via lifecycle policies. Automatically moves infrequently accessed objects to cheaper storage tiers.

#### 5. CloudWatch Log Retention

```hcl
# In terraform.tfvars
log_retention_days = 7  # Instead of 30
```

Savings: ~$2/month

#### 6. Development Environment

For dev/testing:

```hcl
environment               = "dev"
rds_instance_class        = "db.t3.micro"  # Free tier eligible
api_desired_count         = 1
rds_multi_az              = false
log_retention_days        = 7
use_spot_instances        = true
```

**Dev environment cost: ~$50/month**

---

## Rollback Procedure

### Rollback ECS Service

```bash
# List task definition revisions
aws ecs list-task-definitions \
  --family-prefix fraud-detector-api \
  --sort DESC

# Update service to previous revision
aws ecs update-service \
  --cluster mlops-fraud-production \
  --service fraud-detector-api-service \
  --task-definition fraud-detector-api:PREVIOUS_REVISION
```

### Rollback via Terraform

```bash
# Revert to previous commit
git log --oneline
git checkout <previous-commit-sha> deployment/aws/terraform/

# Apply previous state
terraform plan
terraform apply
```

### Rollback Database

```bash
# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier mlops-fraud-production-rds-restored \
  --db-snapshot-identifier <snapshot-id>

# Update connection strings
```

---

## Destroy Infrastructure

**⚠️ WARNING: This will delete all resources and data!**

```bash
# Disable deletion protection first
aws rds modify-db-instance \
  --db-instance-identifier mlops-fraud-production-rds \
  --no-deletion-protection

# Destroy with Terraform
cd deployment/aws/terraform
terraform destroy

# Confirm by typing 'yes'
```

**Manual cleanup** (if needed):
```bash
# Delete S3 buckets (must be empty)
aws s3 rb s3://mlops-fraud-production-data --force
aws s3 rb s3://mlops-fraud-production-artifacts --force

# Delete ECR images
aws ecr batch-delete-image \
  --repository-name fraud-detector-api \
  --image-ids imageTag=latest
```

---

## Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLflow Deployment](https://www.mlflow.org/docs/latest/tracking.html#tracking-server)

---

**Need Help?**
- Check CloudWatch Logs
- Review ECS service events
- Contact: your-email@example.com
