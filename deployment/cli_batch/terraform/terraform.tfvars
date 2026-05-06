project_id            = "dassa-lab"
region                = "asia-east1"
vertex_location       = "us-central1"
bucket_root           = "gs://alphacouncil"
market                = "tw"
masters               = "1,2,3"
report_format         = "json"
timeout_seconds       = 1800
schedule              = "10 16 * * 1-5"
time_zone             = "Asia/Taipei"
job_memory            = "4Gi"
job_cpu               = "2"
poll_interval_seconds = 10

tickers = [
  "2330",
  "2308",
  "2454",
  "2317",
  "3711",
  "2383",
  "2345",
  "3037",
  "2303",
  "2382",
  "2881",
  "2891",
  "2882",
  "2886",
  "2327",
]
