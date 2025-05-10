terraform {
  backend "s3" {
    bucket = "dev-tfstate-aws-iac-book-project-hk-ch6"
    key    = "vpc/terraform.tfstate"
    region = "ap-northeast-1"
  }
}
