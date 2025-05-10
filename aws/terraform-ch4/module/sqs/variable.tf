variable "stage" {
    type        = string
    description = "環境名"
}
variable "queue_name_suffix" {
    type        = string
    description = "SQSのキュー名の接尾辞"  
}
variable "sqs_queue_visibility_timeout_seconds" {
    type        = number
    description = "SQSキューのメッセージの可視性タイムアウト"
}