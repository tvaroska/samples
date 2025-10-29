data "google_client_config" "this" {}


resource "google_vertex_ai_reasoning_engine" "reasoning_engine" {
  display_name = "Terraform"
  project      = data.google_client_config.this.project
  description  = "A basic reasoning engine"
  region       = "us-central1"

  spec {
    agent_framework = "google-adk"

    package_spec {
      dependency_files_gcs_uri = "gs://${data.google_client_config.this.project}/terraform/empty.tar.gz"
      pickle_object_gcs_uri    = "gs://${data.google_client_config.this.project}/terraform/agent.pkl"
      python_version           = "3.12"
      requirements_gcs_uri     = "gs://${data.google_client_config.this.project}/terraform/requirements.txt"
    }
  }
}
