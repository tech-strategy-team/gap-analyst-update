# Terraform VPC

## Terraform Registry

[registry.terraform.io](https://registry.terraform.io)

VPCには多くのリソース配置が必要になる。Terraformの標準機能で記述する場合、大量のresourceブロックを記述する必要があり、大変になる。Trraform Registryでは、複数のリソースをまとめて記述したモジュールが公開されており、ルートモジュールや子モジュールから呼び出して使うことができる。

## AWS VPC Terraform Module

[terraform-aws-modules](https://registry.terraform.io/modules/terraform-aws-modules/vpc/aws/latest)
