variable "stage" {
    type        = string
    description = "stage; dev, prd"
}
variable "vpc_cidr" {
    type        = string
    description = "CIDR for VPC"
}
variable "enable_nat_gateway" {
    type        = bool
    description = "Enable NAT Gateway"
}
variable "one_nat_gateway_per_az" {
    type        = bool
    default     = false
    description = "One NAT Gateway per AZ"  
}
