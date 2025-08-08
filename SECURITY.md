# Security Documentation

## 🔒 Security Overview

This document outlines the security measures implemented in the Enhanced Multi-Agent Stock Analysis System to protect sensitive data and API credentials.

## 🚨 Critical Security Measures

### 1. Environment Variables
- **All API keys are stored in environment variables**
- **No hardcoded credentials in source code**
- **`.env` file is excluded from version control**

### 2. Protected Files
The following files contain sensitive information and are **NEVER committed to version control**:
- `.env` - Environment variables with API keys
- `credentials.json` - Google API credentials
- `token.json` - Google API tokens
- `*.key` - Any key files
- `*.pem` - Certificate files
- `*.p12` - Certificate files

### 3. API Key Management

#### OpenAI API
- **Environment Variable**: `OPENAI_API_KEY`
- **Usage**: All OpenAI API calls use `os.getenv('OPENAI_API_KEY')`
- **Security**: Keys are never logged or printed

#### Notion API
- **Environment Variable**: `NOTION_TOKEN`
- **Usage**: Notion client initialization uses `os.getenv('NOTION_TOKEN')`
- **Security**: Tokens are never exposed in logs

#### Azure OpenAI (Alternative)
- **Environment Variables**: 
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_ENDPOINT`
  - `AZURE_MODEL_NAME`
  - `AZURE_API_VERSION`

## 🛡️ Security Best Practices Implemented

### 1. Input Validation
- All user inputs are validated before processing
- No direct SQL injection possible (using parameterized queries)
- File path validation to prevent directory traversal

### 2. Error Handling
- Sensitive information is never exposed in error messages
- Generic error messages for security-related failures
- Proper exception handling without exposing internal details

### 3. Logging Security
- API keys are never logged
- Sensitive data is masked in logs
- Usage tracking only logs non-sensitive metrics

### 4. File System Security
- Secure file permissions
- Temporary files are properly cleaned up
- Screenshots and data files are stored securely

## 🔧 Security Configuration

### Environment Setup
```bash
# Create .env file with your actual credentials
cp .env.example .env
# Edit .env with your actual API keys
```

### Required Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY="your-actual-openai-api-key"

# Notion Configuration  
NOTION_TOKEN="your-actual-notion-token"
NOTION_STOCK_LIST_PAGE_ID="your-actual-page-id"
NOTION_REPORTS_PAGE_ID="your-actual-page-id"

# Optional Azure Configuration
AZURE_OPENAI_API_KEY="your-azure-key"
AZURE_ENDPOINT="your-azure-endpoint"
AZURE_MODEL_NAME="your-model-name"
AZURE_API_VERSION="2024-12-01-preview"
```

## 🚫 Security Restrictions

### What's NOT Allowed
1. **Hardcoded API keys** in source code
2. **Committing `.env` files** to version control
3. **Logging sensitive credentials**
4. **Exposing API keys** in error messages
5. **Storing credentials** in plain text files

### What's Protected
1. **API keys** are environment variables only
2. **Database credentials** are environment variables
3. **User data** is encrypted in transit
4. **Log files** contain no sensitive information
5. **Error messages** are generic and safe

## 🔍 Security Audit Checklist

Before deploying or sharing code:

- [ ] No hardcoded API keys in source code
- [ ] `.env` file is in `.gitignore`
- [ ] `credentials.json` and `token.json` are in `.gitignore`
- [ ] All API calls use environment variables
- [ ] No sensitive data in logs
- [ ] Error messages don't expose internal details
- [ ] File permissions are secure
- [ ] Temporary files are cleaned up

## 🚨 Incident Response

### If API Keys are Compromised
1. **Immediately rotate** the compromised API keys
2. **Check usage logs** for unauthorized access
3. **Review recent commits** for accidental exposure
4. **Update environment variables** with new keys
5. **Monitor for suspicious activity**

### Reporting Security Issues
If you discover a security vulnerability:
1. **Do not create public issues** for security problems
2. **Contact the maintainer** privately
3. **Provide detailed information** about the vulnerability
4. **Allow time for investigation** and fix

## 📋 Security Compliance

This system follows security best practices:
- ✅ **OWASP Top 10** compliance
- ✅ **Environment variable** usage for secrets
- ✅ **Input validation** and sanitization
- ✅ **Secure error handling**
- ✅ **Proper file permissions**
- ✅ **No credential logging**

## 🔐 Additional Security Recommendations

### For Production Deployment
1. **Use a secrets management service** (AWS Secrets Manager, Azure Key Vault, etc.)
2. **Implement rate limiting** for API calls
3. **Add monitoring and alerting** for unusual activity
4. **Regular security audits** of dependencies
5. **Keep dependencies updated** to latest secure versions

### For Development
1. **Use different API keys** for development and production
2. **Never commit** test credentials
3. **Use mock services** for testing when possible
4. **Regular security reviews** of code changes

---

**Remember**: Security is everyone's responsibility. Always follow these guidelines to keep the system and data secure. 