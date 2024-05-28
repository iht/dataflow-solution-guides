#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

// Project
module "google_cloud_project" {
  source          = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/project?ref=v30.0.0"
  billing_account = var.billing_account
  project_create  = var.project_create
  name            = var.project_id
  parent          = var.organization
  services        = [
    "dataflow.googleapis.com",
    "monitoring.googleapis.com",
    "pubsub.googleapis.com",
    "autoscaling.googleapis.com",
    "spanner.googleapis.com",
    "bigquery.googleapis.com"
  ]
}

// Buckets for staging data, scripts, etc, in the two regions
module "buckets" {
  source        = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/gcs?ref=v30.0.0"
  project_id    = module.google_cloud_project.project_id
  name          = module.google_cloud_project.project_id
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = var.destroy_all_resources
}

// BigQuery dataset for final destination
module "dataset" {
  source     = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/bigquery-dataset?ref=v30.0.0"
  project_id = module.google_cloud_project.project_id
  id         = "replica"
  access = {
    dataflow-writer = { role = "OWNER", type = "user" }
  }
  access_identities = {
    dataflow-writer = module.dataflow_sa.email
  }
}

// Spanner instance for change streams / CDC
resource "google_spanner_instance" "spanner_instance" {
  config        = "regional-${var.region}"
  project       = module.google_cloud_project.project_id
  display_name  = "Test Spanner Instance"
  processing_units = 100
  // This is the minimal instance, you will want to change this for production deployments.
  force_destroy = var.destroy_all_resources
}

resource "google_spanner_database" "taxis" {
  instance = google_spanner_instance.spanner_instance.name
  project  = module.google_cloud_project.project_id
  name     = "taxis_database"
  ddl      = [
    <<DDL
CREATE TABLE events (
  ride_id STRING(64),
  point_idx INT64,
  latitude FLOAT64,
  longitude FLOAT64,
  timestamp TIMESTAMP,
  meter_reading FLOAT64,
  meter_increment FLOAT64,
  ride_status STRING(64),
  passenger_count INT64,
) PRIMARY KEY(ride_id, point_idx)
DDL
  ]
  deletion_protection = !var.destroy_all_resources
}

resource "google_spanner_database" "metadata" {
  instance            = google_spanner_instance.spanner_instance.name
  project             = module.google_cloud_project.project_id
  name                = "metadata"
  deletion_protection = !var.destroy_all_resources
}

// Service account
module "dataflow_sa" {
  source       = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/iam-service-account?ref=v30.0.0"
  project_id   = module.google_cloud_project.project_id
  name         = "my-dataflow-sa"
  generate_key = false
  iam_project_roles = {
    (module.google_cloud_project.project_id) = [
      "roles/storage.admin",
      "roles/dataflow.worker",
      "roles/monitoring.metricWriter",
      "roles/pubsub.editor"
    ]
  }
}

// Network
module "vpc_network" {
  source     = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/net-vpc?ref=v30.0.0"
  project_id = module.google_cloud_project.project_id
  name       = "default"
  subnets    = [
    {
      ip_cidr_range         = "10.1.0.0/16"
      name                  = "default"
      region                = var.region
      enable_private_access = true
      secondary_ip_ranges = {
        pods     = "10.16.0.0/14"
        services = "10.20.0.0/24"
      }
    }
  ]
}

module "firewall_rules" {
  // Default rules for internal traffic + SSH access via IAP
  source     = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/net-vpc-firewall?ref=v30.0.0"
  project_id = module.google_cloud_project.project_id
  network    = module.vpc_network.name
  default_rules_config = {
    admin_ranges = [
      module.vpc_network.subnet_ips["${var.region}/default"],
    ]
  }
  egress_rules = {
    allow-egress-dataflow = {
      deny        = false
      description = "Dataflow firewall rule egress"
      targets     = ["dataflow"]
      rules       = [{ protocol = "tcp", ports = [12345, 12346] }]
    }
  }
  ingress_rules = {
    allow-ingress-dataflow = {
      description = "Dataflow firewall rule ingress"
      targets     = ["dataflow"]
      rules       = [{ protocol = "tcp", ports = [12345, 12346] }]
    }
  }
}

// So we can get to Internet if necessary (from the Dataflow region)
module "regional_nat" {
  count          = var.internet_access ? 1 : 0
  source         = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/net-cloudnat?ref=v30.0.0"
  project_id     = module.google_cloud_project.project_id
  region         = var.region
  name           = "default"
  router_network = module.vpc_network.self_link
}