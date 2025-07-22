# AWS Amplify Deployment Setup Guide

## Environment Variables Configuration

### Required Environment Variables (Set in Amplify Console)

Navigate to: **AWS Amplify Console** → **Your App** → **App settings** → **Environment variables**

Add the following variables:

| Variable Name | Value | Description |
|---------------|-------|-------------|
| `OPENAI_API_KEY` | `sk-proj-your-actual-key` | OpenAI API key for AI functionality |
| `LANGCHAIN_API_KEY` | `your-langchain-key` | LangChain API key |
| `LANGCHAIN_TRACING_V2` | `false` | LangChain tracing setting |
| `LANGCHAIN_PROJECT` | `motor-parts-api` | LangChain project name |
| `SERVER_HOSTNAME` | `dbc-73ecee4a-e5cc.cloud.databricks.com` | Databricks server hostname |
| `HTTP_PATH` | `/sql/1.0/warehouses/6d66dae205b7527d` | Databricks HTTP path |
| `ACCESS_TOKEN` | `your-databricks-token` | Databricks access token |
| `AWS_ACCESS_KEY_ID` | `your-aws-key` | AWS access key (if needed) |
| `AWS_SECRET_ACCESS_KEY` | `your-aws-secret` | AWS secret key (if needed) |
| `AWS_SESSION_TOKEN` | `your-aws-session-token` | AWS session token (if needed) |

## Security Best Practices

### ✅ DO:
- Set sensitive values in Amplify Console Environment Variables
- Use the `.env.template` file to document required variables
- Keep `.env` files in `.gitignore`
- Rotate API keys regularly

### ❌ DON'T:
- Commit `.env` files to version control
- Hardcode API keys in source code
- Share API keys in chat/email
- Use production keys in development

## Deployment Steps

1. **Push Code to GitHub**
   ```bash
   git add .
   git commit -m "Updated configuration"
   git push origin main
   ```

2. **Configure Amplify**
   - Connect GitHub repository
   - Set environment variables in console
   - Choose `amplify.yml` as build configuration

3. **Deploy**
   - Amplify will automatically build and deploy
   - Environment variables are injected at build time
   - Application will have access to all configured variables

## Verification

After deployment, check that:
- [ ] CSS and styling loads properly
- [ ] API endpoints respond correctly
- [ ] Database connections work
- [ ] Part number processing functions correctly
- [ ] No API keys are visible in browser/logs

## Troubleshooting

### Common Issues:
1. **CSS not loading**: Check file paths in `index.html`
2. **API errors**: Verify environment variables in Amplify Console
3. **Database connection**: Check Databricks credentials
4. **Build failures**: Review build logs in Amplify Console

### Build Logs Location:
AWS Amplify Console → Your App → Deployment → View build logs
