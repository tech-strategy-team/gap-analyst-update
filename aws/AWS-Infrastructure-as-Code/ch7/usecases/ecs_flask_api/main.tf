resource "aws_ecs_cluster" "flask_api" {
    name = "${var.stage}-flask-api-tf"
}
resource "aws_ecs_cluster_capacity_providers" "flask_api" {
    cluster_name = aws_ecs_cluster.flask_api.name
    capacity_providers = [
        "FARGATE",
    ]
}

data "aws_iam_policy_document" "ecs_task_execution_assume_role" {
    statement {
        effect = "Allow"
        actions = [
            "sts:AssumeRole",
        ]
        principals {
            type        = "Service"
            identifiers = ["ecs-tasks.amazonaws.com"]
        }
    }
}

