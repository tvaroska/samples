## Example of chatbot on Cloud Run

### Deploy
1. Edit .env -> modify NEXT_PUBLIC_PROJECT variable
2. gcloud run deploy <service name>
3. gcloud run services proxy <service name>