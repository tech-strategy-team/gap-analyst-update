# CloudFormationでWebシステムを構築する

[Hands-on-for-Beginners-Scalable](https://pages.awscloud.com/event_JAPAN_Hands-on-for-Beginners-Scalable_LP.html)

- [AWS Well Architected フレームワーク](https://aws.amazon.com/jp/architecture/well-architected/)
- [AWS Black Belt Online Seminar AWS CloudFormation](https://www.slideshare.net/AmazonWebServicesJapan/aws-black-belt-online-seminar-aws-cloudformation)

## Doc

- [CloudFormation テンプレートリファレンス](https://docs.aws.amazon.com/ja_jp/AWSCloudFormation/latest/UserGuide/template-reference.html)
- [CloudFormation リソースタイプ](https://docs.aws.amazon.com/ja_jp/AWSCloudFormation/latest/UserGuide/aws-template-resource-type-ref.html)
- [CloudFormation リリース履歴](https://docs.aws.amazon.com/ja_jp/AWSCloudFormation/latest/UserGuide/ReleaseHistory.html)
- [CloudFormation サンプルテンプレート](https://docs.aws.amazon.com/ja_jp/AWSCloudFormation/latest/UserGuide/cfn-sample-templates.html)
- [CloudFormation スタックの削除に失敗したとき](https://aws.amazon.com/jp/premiumsupport/knowledge-center/cloudformation-stack-delete-failed/)
- [CloudFormation CodePipeline を使用した継続的デリバリー](https://docs.aws.amazon.com/ja_jp/AWSCloudFormation/latest/UserGuide/continuous-delivery-codepipeline.html)

## Validete

```console
aws cloudformation validate-template --template-body file://02_ec2.yml
```

## Create

```console
aws cloudformation update-stack --stack-name handson-cfn --template-body file://01_vpc.yml
aws cloudformation create-stack --stack-name handson-cfn-ec2 --template-body file://02_ec2.yml
aws cloudformation create-stack --stack-name handson-cfn-rds --template-body file://03_rds.yml
aws cloudformation create-stack --stack-name handson-cfn-elb --template-body file://04_elb.yml
```
