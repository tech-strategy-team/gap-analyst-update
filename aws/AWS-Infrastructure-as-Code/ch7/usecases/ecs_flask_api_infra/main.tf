resource "aws_ecs_repository" "flask_api" {
  name = "${var.stage}-flask-api-tf"
}
resource "aws_ssm_parameter" "flask_api_correct_answer" {
    name = "/fask-api-tf/${var.stage}/correct_answer"
    type = "SecureString"
    value = "uninitialized"
    # 格納された値が変更されても無視する
    lifecycle {
        ignore_changes = [
            value
        ]
    }
  
}