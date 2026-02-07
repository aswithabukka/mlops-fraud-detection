# GitHub Branch Protection Setup Guide

Follow these steps to enable branch protection rules for your repository.

## 🔒 Why Branch Protection?

Branch protection ensures:
- Code quality (tests must pass before merge)
- Code review (require PR approvals)
- No accidental force pushes
- Linear history (optional)
- Protected production code

## 📋 Setup Instructions

### Step 1: Navigate to Settings

1. Go to your repository: https://github.com/aswithabukka/mlops-fraud-detection
2. Click **Settings** (top right)
3. Click **Branches** (left sidebar)
4. Click **Add branch protection rule**

### Step 2: Configure Protection Rule

#### Branch Name Pattern
```
main
```

#### Protect Matching Branches - Enable These:

##### ✅ Require a pull request before merging
- [x] **Require a pull request before merging**
  - Number of approvals required: **1**
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners (if you create a CODEOWNERS file)

##### ✅ Require status checks to pass before merging
- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging

  **Select these status checks** (after your first CI run):
  - `lint / Code Quality & Linting`
  - `test-unit / Unit Tests`
  - `test-integration / Integration Tests`
  - `docker-build / Docker Build Test`
  - `security-scan / Security Vulnerability Scan`

##### ✅ Require conversation resolution before merging
- [x] **Require conversation resolution before merging**
  - All PR comments must be resolved before merge

##### ✅ Require signed commits (Optional, but recommended)
- [ ] **Require signed commits**
  - Ensures commits are verified with GPG/SSH keys
  - Skip this if you haven't set up commit signing

##### ✅ Require linear history (Optional)
- [x] **Require linear history**
  - Prevents merge commits, requires rebase or squash
  - Keeps git history clean

##### ✅ Include administrators
- [x] **Include administrators**
  - Apply rules to repository administrators too
  - Ensures no one bypasses protection

##### ❌ Do not allow bypassing
- [x] **Do not allow bypassing the above settings**
  - Even admins can't bypass (strict mode)

##### ⚠️ Restrictions (Optional)
- [ ] **Restrict who can push to matching branches**
  - Leave unchecked for personal project
  - For team: Specify users/teams who can push

##### 🔄 Force Push Protection
- [x] **Do not allow force pushes**
  - Prevents `git push --force`
  - Protects history integrity

##### 🗑️ Deletion Protection
- [x] **Do not allow deletions**
  - Prevents branch deletion
  - Keeps main branch safe

### Step 3: Save Protection Rule

Click **Create** or **Save changes** at the bottom.

## ✅ Verification

After setup, try to:
1. Push directly to `main` → Should fail ❌
2. Create PR without tests passing → Should block merge ❌
3. Create PR with passing tests → Should allow merge after approval ✅

## 🔧 Recommended Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes and commit
git add .
git commit -m "feat: add new feature"

# 3. Push to GitHub
git push origin feature/new-feature

# 4. Create PR on GitHub
# - Fill in PR template
# - Wait for CI checks to pass
# - Request review
# - Address review comments
# - Merge after approval
```

## 📝 Additional Setup (Optional)

### Create CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Default owner for everything
* @aswithabukka

# MLOps components
/airflow/ @aswithabukka
/deployment/ @aswithabukka

# Infrastructure
*.tf @aswithabukka
*.yml @aswithabukka
```

### Create Pull Request Template

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass locally
- [ ] Integration tests pass locally
- [ ] Tested manually

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Added/updated documentation
- [ ] No new warnings
- [ ] Added tests
```

### Create Issue Templates

Create `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md` for standardized issues.

## 🎯 Benefits After Setup

- ✅ Higher code quality (automated checks)
- ✅ Better collaboration (required reviews)
- ✅ Protected history (no force pushes)
- ✅ Clear workflow (PRs only)
- ✅ Audit trail (all changes reviewed)

## ⚠️ Important Notes

1. **First CI Run**: Status checks won't appear until your first GitHub Actions run completes
2. **Testing**: Create a test branch to verify protection works
3. **Flexibility**: You can always update rules later in Settings → Branches
4. **Emergency**: If you need to bypass, temporarily disable protection (not recommended)

## 🆘 Troubleshooting

### "Cannot see status checks"
- Wait for at least one CI run to complete
- Status checks appear after first workflow execution

### "Cannot merge PR even though checks pass"
- Ensure branch is up to date with main
- Check if all conversations are resolved
- Verify you have required approvals

### "Need to make urgent fix"
- Temporarily disable protection
- Make fix
- Re-enable protection immediately
- Better: Create PR and self-approve

## 📚 Further Reading

- [GitHub Branch Protection Docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Required Status Checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging)
- [Code Owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)

---

**Ready to protect your main branch!** 🛡️
