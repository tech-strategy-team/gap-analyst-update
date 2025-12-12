# AWSデプロイガイド

このドキュメントでは、Streamlitアプリを4つの方法でAWSにデプロイする手順を説明します。

## 目次

1. [AWS App Runner（推奨・最も簡単）](#1-aws-app-runner推奨最も簡単)
2. [ECS Fargate（スケーラブル）](#2-ecs-fargateスケーラブル)
3. [EC2（従来型）](#3-ec2従来型)
4. [比較表](#4-デプロイ方法の比較)

---

## 1. AWS App Runner（推奨・最も簡単）

**特徴**: コンテナを自動でビルド・デプロイ・スケール。最も簡単。

### 手順

#### 1.1 GitHubにコードをプッシュ

```bash
git add .
git commit -m "Add Streamlit CSV visualizer"
git push origin main
```

#### 1.2 AWS App Runnerでサービスを作成

1. **AWSマネジメントコンソール**にログイン
2. **App Runner**サービスを開く
3. **「サービスの作成」**をクリック

4. **ソース設定**:
   - ソースコードリポジトリ: GitHub
   - リポジトリ: `n-group`を選択
   - ブランチ: `main`
   - デプロイトリガー: 自動

5. **ビルド設定**:
   - ランタイム: Python 3
   - ビルドコマンド: `pip install -r requirements.txt`
   - 開始コマンド: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`
   - ポート: `8501`

6. **サービス設定**:
   - サービス名: `streamlit-csv-visualizer`
   - vCPU: 1
   - メモリ: 2 GB

7. **作成**をクリック

#### 1.3 デプロイ完了

数分後、App RunnerがURLを提供します。例: `https://xxxxx.awsapprunner.com`

### 料金

- vCPU: $0.064/vCPU-hour
- メモリ: $0.007/GB-hour
- 月額推定: $50-100（常時稼働の場合）

---

## 2. ECS Fargate（スケーラブル）

**特徴**: コンテナオーケストレーション。高スケーラビリティ。

### 手順

#### 2.1 ECR（Elastic Container Registry）にDockerイメージをプッシュ

```bash
# AWSにログイン
aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com

# ECRリポジトリを作成
aws ecr create-repository --repository-name streamlit-csv-visualizer --region ap-northeast-1

# Dockerイメージをビルド
docker build -t streamlit-csv-visualizer .

# イメージにタグ付け
docker tag streamlit-csv-visualizer:latest <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/streamlit-csv-visualizer:latest

# イメージをプッシュ
docker push <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/streamlit-csv-visualizer:latest
```

#### 2.2 ECSクラスタを作成

1. **ECSコンソール**を開く
2. **クラスターの作成**をクリック
3. クラスタ名: `streamlit-cluster`
4. **Fargate**を選択

#### 2.3 タスク定義を作成

1. **タスク定義**→**新しいタスク定義の作成**
2. タスク定義名: `streamlit-task`
3. 起動タイプ: **Fargate**
4. **タスクサイズ**:
   - CPU: 0.5 vCPU
   - メモリ: 1 GB
5. **コンテナ定義**:
   - コンテナ名: `streamlit`
   - イメージURI: `<account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/streamlit-csv-visualizer:latest`
   - ポートマッピング: `8501`

#### 2.4 サービスを作成

1. クラスタを開く
2. **サービスの作成**をクリック
3. 起動タイプ: **Fargate**
4. タスク定義: `streamlit-task`
5. サービス名: `streamlit-service`
6. タスク数: 1
7. **ロードバランサー**:
   - タイプ: Application Load Balancer
   - ターゲットグループ: 新規作成
   - パス: `/`
   - ヘルスチェックパス: `/_stcore/health`

#### 2.5 ALBのDNS名でアクセス

ロードバランサーのDNS名（例: `streamlit-alb-xxxxx.ap-northeast-1.elb.amazonaws.com`）でアクセス可能。

### 料金

- Fargate: vCPU $0.04/hour + メモリ $0.004/GB/hour
- ALB: $0.0225/hour + データ転送
- 月額推定: $30-60

---

## 3. EC2（従来型）

**特徴**: 完全な制御。手動設定が必要。

### 手順

#### 3.1 EC2インスタンスを起動

1. **EC2コンソール**→**インスタンスを起動**
2. AMI: **Amazon Linux 2023**
3. インスタンスタイプ: **t3.small**
4. キーペア: 新規作成または既存を選択
5. セキュリティグループ:
   - SSH (22): 自分のIPのみ
   - カスタムTCP (8501): 0.0.0.0/0（全公開）

#### 3.2 SSHでインスタンスに接続

```bash
ssh -i "your-key.pem" ec2-user@<public-ip>
```

#### 3.3 アプリをセットアップ

```bash
# Pythonとgitをインストール
sudo yum update -y
sudo yum install python3 python3-pip git -y

# リポジトリをクローン
git clone https://github.com/your-username/n-group.git
cd n-group/python/streamlit-app

# 仮想環境を作成
python3 -m venv .venv
source .venv/bin/activate

# 依存パッケージをインストール
pip install -r requirements.txt

# Streamlitを起動
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

#### 3.4 systemdでサービス化（自動起動）

```bash
sudo nano /etc/systemd/system/streamlit.service
```

以下を記述:

```ini
[Unit]
Description=Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/n-group/python/streamlit-app
Environment="PATH=/home/ec2-user/n-group/python/streamlit-app/.venv/bin"
ExecStart=/home/ec2-user/n-group/python/streamlit-app/.venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

サービスを有効化:

```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
```

#### 3.5 アクセス

ブラウザで `http://<public-ip>:8501` にアクセス。

### 料金

- t3.small: $0.0208/hour
- 月額推定: $15-20

---

## 4. デプロイ方法の比較

| 項目 | App Runner | ECS Fargate | EC2 |
|------|-----------|-------------|-----|
| **難易度** | ⭐ 最も簡単 | ⭐⭐ やや難しい | ⭐⭐⭐ 難しい |
| **セットアップ時間** | 5分 | 30分 | 1時間 |
| **スケーラビリティ** | 自動 | 自動（ALB必要） | 手動 |
| **メンテナンス** | 不要 | 低 | 高 |
| **料金（月額）** | $50-100 | $30-60 | $15-20 |
| **カスタマイズ性** | 低 | 高 | 最高 |
| **推奨用途** | 小規模・個人 | 中規模・本番 | 実験・学習 |

---

## 5. 推奨デプロイフロー

### 初心者・個人プロジェクト
1. **App Runner** - 最も簡単で管理不要

### 本番環境・スケールが必要
1. **ECS Fargate** - 高可用性とスケーラビリティ

### 学習・実験
1. **EC2** - AWSの基礎を学べる

---

## 6. セキュリティのベストプラクティス

### 6.1 環境変数の管理

機密情報は環境変数で管理:

```bash
# .streamlit/secrets.toml（Gitにコミットしない）
[secrets]
api_key = "your-secret-key"
```

AWS Systems Manager Parameter Storeを使用:

```python
import boto3

ssm = boto3.client('ssm', region_name='ap-northeast-1')
api_key = ssm.get_parameter(Name='/streamlit/api_key', WithDecryption=True)['Parameter']['Value']
```

### 6.2 HTTPS化

- **App Runner**: 自動でHTTPS
- **ECS/EC2**: ALBでSSL証明書を設定（AWS Certificate Manager）

### 6.3 認証

Streamlitアプリに認証を追加:

```python
import streamlit_authenticator as stauth
```

または、AWS Cognitoと連携。

---

## 7. コスト最適化

### 7.1 App Runnerのスケジューリング

夜間停止でコスト削減（手動停止/起動、またはLambdaで自動化）

### 7.2 ECS Fargateのオートスケーリング

トラフィックに応じてタスク数を調整:

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/streamlit-cluster/streamlit-service \
  --min-capacity 1 \
  --max-capacity 5
```

### 7.3 EC2のスポットインスタンス

通常より最大90%安い（ただし中断の可能性あり）

---

## 8. モニタリング

### CloudWatchでログを確認

- **App Runner**: 自動的にCloudWatch Logsに送信
- **ECS**: タスク定義でログドライバーを設定
- **EC2**: CloudWatch Agentをインストール

---

## まとめ

**最も簡単**: AWS App Runner
**本番環境**: ECS Fargate + ALB
**学習・実験**: EC2

質問があれば、お気軽にお尋ねください！
