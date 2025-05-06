provider "aws" {
    region = "ap-northeast-1"
    default_tags {
        tags = {
            Terraform = "true"
            STAGE    = "dev"
            MODULE  = "case1"
        }
    }
}
provider "aws" {
    alias  = "us-east-1"
    region = "us-east-1"
    default_tags {
        tags = {
            Terraform = "true"
            STAGE    = "dev"
            MODULE  = "case1"
        }
    } 
}