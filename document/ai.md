# Artificial Intelligence

## AIとは何か？

### AIの概念の萌芽

#### 機械が知能を持てるか？

<img src="https://wired.jp/app/uploads/2020/05/19c5f75d150a7e0363d63e1062bba3e5.jpg" style="zoom:50%;" />

1950年、アラン・チューリングは論文「計算する機械と知能 (Computing Machinery and Intelligence)」を発表し、**「機械が知能を持てるか？」**という問いを立てました。この論文では、後に「チューリングテスト」と呼ばれる基準を提案し、機械が人間のように振る舞えるかどうかを判断する基準を示しました。

#### チューリングテスト

チューリングテストは、コンピュータが「人間らしさ」をどれだけ表現できるかを評価するもので、具体的には、審査員がコンピュータと人間の対話者を区別できなければ、そのコンピュータは「知的」と見なされるとされました。

![](https://miro.medium.com/v2/resize:fit:758/1*dDv4ExVNwrY-IyaqnmzKcQ.png)

人間の審査員が、見えない相手（人間またはコンピュータ）と会話をし、その会話が人間のように自然であれば合格。

### 「人工知能」という言葉の誕生

#### ダートマス会議

![](https://cdn-ak.f.st-hatena.com/images/fotolife/z/zawapython/20190226/20190226151651.jpg)

人工知能という用語は、この会議で初めて正式に使用されました。この名称は、ジョン・マッカーシーが、研究内容を簡潔かつ的確に表現するために提案したものです。当時、計算機科学や認知科学の研究者たちは、機械がどのようにして人間の知能を模倣できるかについて議論を重ねており、この言葉が学際的な研究の旗印として適していると判断されました。この会議はジョン・マッカーシー、マービン・ミンスキー、クロード・シャノン、ナサニエル・ロチェスターらによって組織され、ダートマス大学で開催されました。会議の目的は、機械が人間のような学習、推論、問題解決などを行える可能性を探ることでした。マッカーシーは、この分野の研究に「人工知能 (Artificial Intelligence)」という名前を付けました。

### AI冬の時代

人工知能研究は1950年代後半から1960年代にかけて大きな期待を集めましたが、その後、いくつかの技術的・社会的要因によって進展が停滞する時期を迎えました。この現象は「AI冬の時代」と呼ばれています。

#### 技術的限界

- **シンボリックAIの限界**: 初期のAIは、ルールベースのシステムや論理的推論に依存していましたが、これらの方法は現実世界の曖昧さや膨大なデータに対応できませんでした。例えば、チェスのような特定の問題には効果的でしたが、曖昧な質問や複雑な自然言語に対応することはできませんでした。
- **自然言語処理の課題**: 当時の自然言語処理（NLP）の技術は、主に文法規則や辞書ベースの手法に依存していました。これにより、文の構造を解析することは可能でしたが、文脈や多義語の意味を正確に理解することができませんでした。例えば、"bank"という単語が「銀行」なのか「川岸」なのかを文脈から判断する能力はほとんどなく、翻訳や対話システムでは誤った解釈が頻繁に発生しました。また、言語モデルが単純なルールに基づいていたため、文法的に正しいが意味的には不自然な出力が生成されることも多かったのです。
- **専門システムの制約**: 専門システム（エキスパートシステム）は、特定の分野に関する知識をルールベースでプログラム化し、専門家のように問題を解決するAIシステムです。例えば、MYCINは感染症の診断に特化したシステムで、症状や検査結果に基づいて適切な診断や治療法を提案するものでした。しかし、このようなシステムは、ルールが固定的であるため、新しい分野や状況に適応することが難しく、汎用性に乏しいという課題がありました。

#### 社会的要因

- **過剰な期待**: 初期のAI研究者たちは、数年以内に人間のように考える機械を実現できると主張しましたが、これらの目標は実現されず、資金提供者の失望を招きました。
- **資金不足**: 実験やプロジェクトが期待通りの成果を上げられなかったため、研究費が削減され、研究活動が縮小しました。例えば、アメリカではDARPAがAI研究への資金提供を大幅に減らしました。

#### 再興への兆し

- **専門システム**: 1970年代後半には、医療や化学などの専門分野で実用的なAIアプリケーションが開発され、再び注目を集めるようになりました。例えば、MYCINは感染症の診断支援に使用されました。
- **日本の第五世代コンピュータ計画**: 1980年代に日本政府が主導したこのプロジェクトは、AI研究の再興を後押ししました。この計画は、並列処理や論理プログラミングを活用した高度なコンピュータの開発を目指しました。
- **検索アルゴリズム**: 情報検索やデータベース技術の向上により、AIは現実的な応用へと活用されるようになりました。例えば、インターネット黎明期には検索エンジンの技術が急速に進化しました。
- **プランニングとエージェント**: 自律的に問題解決や意思決定を行うアルゴリズムの研究が進み、ロボットや自律システムに応用され始めました。例えば、ロボットが自動的にタスクを計画して実行する技術が注目を集めました。

このように、AI冬の時代は単なる停滞期ではなく、基礎技術の課題と可能性を見直す重要な期間でもありました。

## Neural Net

![A biological and an artificial neuron](https://miro.medium.com/v2/resize:fit:610/1*SJPacPhP4KDEB1AdhOFy_Q.png)

ニューラルネットワークは、人工知能の礎となる技術であり、人間の脳神経系の働きを模倣した計算モデルです。その基本的な構造は「ニューロン」と呼ばれる単位が多数連結され、情報を伝達・処理する仕組みから成り立っています。この構造は、複雑なパターンを学習し、未知のデータに対して予測や推論を行う能力を備えています。

- [The differences between Artificial and Biological Neural Networks](https://towardsdatascience.com/the-differences-between-artificial-and-biological-neural-networks-a8b46db828b7)

### Perceptron

![](https://cdn-ak.f.st-hatena.com/images/fotolife/k/kakts/20170102/20170102000614.png)

**概要**

パーセプトロンは、1958年にフランク・ローゼンブラット（Frank Rosenblatt）によって提案された、単純なニューラルネットワークモデルです。このモデルは、入力層と出力層を持ち、入力データに重みをかけた合計値をもとに2値の出力を生成します。数学的には、線形関数を使ってデータを分類するシンプルなアルゴリズムです。

**背景**

1950年代後半、人工知能研究が黎明期を迎える中、機械が「学習」し、パターンを認識できる能力が求められていました。パーセプトロンは、このような期待に応える最初のモデルとして開発されました。

**特徴**

パーセプトロンは、簡単なパターン認識や分類タスクを解決することができました。例えば、手書き文字の識別やシンプルな形状の分類といった問題に応用され、従来の固定ルールに基づいたプログラムでは対応できなかった柔軟性を示しました。これは機械が自動的に「データから学ぶ」仕組みの可能性を初めて証明したと言えます。

**重要性**

パーセプトロンは、機械学習の基本概念である「学習」の実現可能性を初めて示しました。特に、簡単なパターン認識タスクにおいて成果を上げ、AI研究への期待を高めました。

**課題**

パーセプトロンには重大な限界がありました。それは、線形分離可能なデータしか扱えない点です。たとえば、論理演算の「XOR」問題を解決できないことが、マービン・ミンスキーとシーモア・パパートの著書『パーセプトロン』で指摘され、研究の進展にブレーキをかけました。

### Multi-Layer Perceptron

![](https://www.researchgate.net/publication/303875065/figure/fig4/AS:371118507610123@1465492955561/A-hypothetical-example-of-Multilayer-Perceptron-Network.png)

**概要**

マルチレイヤーパーセプトロン（MLP）は、ニューラルネットワークの基本形であり、複数の**隠れ層（hidden layers）を持つ全結合型のネットワークです。このモデルは、各層が入力データを処理し、次の層に伝達することで、データの複雑な非線形パターンを学習する能力を持ちます。特に、隠れ層と非線形活性化関数**（ReLUやシグモイド関数など）を導入することで、単層パーセプトロンの限界（例: XOR問題）を克服しました。

**背景**

MLPは、パーセプトロンの限界を指摘したマービン・ミンスキーとシーモア・パパートの著書『Perceptrons』（1969年）における問題を克服するために開発されました。彼らは、単層パーセプトロンでは線形分離不可能な問題（XOR問題）を解決できないことを示しました。1970年代から1980年代にかけて、計算能力の向上や理論的ブレイクスルー（例: バックプロパゲーションアルゴリズム）により、MLPが研究されるようになりました。

**特徴**

MLPは、以下のような非線形な問題を解決できるようになりました：

- **XOR問題**: 単層パーセプトロンでは解けなかった線形分離不可能な問題を解決可能。
- **パターン認識**: 手書き文字認識、音声認識、画像分類などの基本的なパターン認識タスク。
- **関数近似**: 非線形関数の近似により、様々な入力と出力の関係を学習可能。

これにより、より複雑なタスクに対してもニューラルネットワークの適用が可能になりました。

**課題・限界**

- **計算負荷の高さ**: 隠れ層が増えると計算コストが増大。特に、1980年代には計算資源が十分でなかった。
- **データの不足**: MLPは多くのデータを必要とするが、当時は大規模データセットが限られていた。
- **局所最適解**: 学習時に局所最適解に陥る可能性があり、最適なパフォーマンスを保証できない場合がある。

### Backpropagation

![](https://media.geeksforgeeks.org/wp-content/uploads/20240217152156/Frame-13.png)

**概要**

バックプロパゲーション（Backpropagation）は、ニューラルネットワークの重みを効率的に学習するためのアルゴリズムです。ネットワークの出力と目標値との間の誤差を計算し、その誤差を逆方向に伝播させることで、各層の重みを更新します。このプロセスは、**勾配降下法（Gradient Descent）**を用いてネットワーク全体の誤差を最小化することを目的としています。

**背景**

バックプロパゲーションは、ニューラルネットワークの学習プロセスを革新した技術ですが、その基礎的なアイデアは1970年代にさかのぼります。初期の研究では、ニューラルネットの重み調整が非常に非効率であったため、大規模なネットワークを学習させることが困難でした。

**特徴**

- **多層ニューラルネットワークの学習**: 従来の手法では、隠れ層の重みを効果的に調整できませんでしたが、バックプロパゲーションにより多層構造が実用化。
- **学習効率の向上**: 勾配降下法を活用することで、誤差の収束が迅速に行えるようになりました。
- **非線形問題の解決**: 非線形活性化関数（シグモイド関数など）と組み合わせることで、複雑なデータ構造を学習可能にしました。

**課題・限界**

- **勾配消失問題**: 隠れ層が多い場合、誤差が伝播する際に勾配が極端に小さくなる問題（勾配消失）が発生。
- **局所最適解**: 非線形関数を持つネットワークでは、誤差関数が局所的最適解に陥る可能性がある。
- **計算コスト**: ネットワークが大規模になると計算負荷が高くなる。

## Deep Learning

### Convolutional Neural Network

畳み込みニューラルネットワーク

#### 1998年 LeNet

![LeNet-5 architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/LeNet-5_architecture.svg/1599px-LeNet-5_architecture.svg.png)

ルネット（LeNet）は、1990年代初頭にヤン・ルカン（Yann LeCun）によって提案された、畳み込みニューラルネットワーク（Convolutional Neural Network, CNN）の初期モデルです。このモデルは、画像データから局所的な特徴を効率的に学習するために、**畳み込み層**と**プーリング層**を導入しました。さらに、これらの特徴を分類するために、全結合層を組み合わせた構造を持っています。特に手書き数字認識（MNISTデータセット）において優れた成果を上げました。

- [Gradient-based learning applied to document recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

#### 2012年 AlexNet

![AlexNet-Architecture](https://miro.medium.com/v2/resize:fit:1400/1*0dsWFuc0pDmcAmHJUh7wqg.png)

[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

**画像認識技術の課題**

- 画像をうまく分類できなかった
  - コンピューターにとって、画像を認識することは「ピクセルの配列から特徴を見つける作業」です。しかし、従来の方法では手作業で「特徴」を定義する必要がありました（例えば、エッジや角の検出など）
  - 画像の数が増えると、計算量が増大し、性能も頭打ちになっていました。
- データが大きくても使いこなせなかった
  - 画像データセット（例えばMNISTやCIFAR-10）はサイズが小さく、大規模なモデルの学習には不十分でした。
  - 高性能なネットワークを学習させるには、膨大な計算が必要です。しかし、当時のCPUでは時間がかかりすぎて実用的ではありませんでした。

**AlexNetが解決したこと**

- 画像を分類できる＝**深いネットワーク構造（Deep Convolutional Neural Networks）**

​	•	AlexNetは8層のネットワークを持つモデルです：

​	•	**5つの畳み込み層（Convolutional Layers）**：画像から特徴を抽出。

​	•	**3つの全結合層（Fully Connected Layers）**：抽出した特徴を基に分類を行う。

​	•	最後は**Softmax層**を使用し、1000クラスへの分類確率を出力。

​	•	この「深さ」が、高度な特徴を学習する鍵となりました。

**GPUを活用した高速学習**

​	•	GPUは画像処理に特化した並列計算が得意です。AlexNetではNVIDIA GTX 580を2枚使用し、並列計算を最適化しました。

​	•	ネットワークの一部を各GPUに割り振り、効率的にトレーニングを行いました。

訓練データのバリエーションを増やすために以下を行いました：

​	•	ランダムに画像の一部を切り取ったり、左右反転させる。

​	•	RGBチャンネルの輝度をランダムに変更する。

​	•	これにより、モデルがさまざまな画像に対応できるようになりました。

### Generative Adversarial Nets

#### 2014年　Generative Adversarial Nets.

### Reinforcement Learning 

<iframe width="560" height="315" src="https://www.youtube.com/embed/TmPfTpjtdgg?si=8kN9efSwzSETyRUG" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[Deep Reinforcement Learning](https://deepmind.google/discover/blog/deep-reinforcement-learning/)

#### 2016年 DeepMinad Alpha Go

![](https://cdn.packtpub.com/article-hub/articles/40f4e78d2c7769d6840959fc99c50288.png)

[AlphaGo](https://deepmind.google/research/breakthroughs/alphago/)

### Transformer

#### 2017年 Google Attention Is All You Need

![スクリーンショット 2024-12-08 15.49.49](/Users/kouichihara/Library/Application Support/typora-user-images/スクリーンショット 2024-12-08 15.49.49.png)

[論文解説 Attention Is All You Need (Transformer)](https://deeplearning.hatenablog.com/entry/transformer)

GPT-1

#### 2019年 OpenAI GPT-2

![スクリーンショット 2024-12-08 16.48.46](/Users/kouichihara/Library/Application Support/typora-user-images/スクリーンショット 2024-12-08 16.48.46.png)

[Better language models and their implications](https://openai.com/index/better-language-models/)

**モデルの大規模化と性能向上:** GPT-2は、事前学習モデルの一種で、従来よりも遥かに大きなモデル（15億パラメータ）と幅広いデータを用いたことで、高度な言語理解・生成性能を示しました。

**驚くべき生成能力:** 当時の水準からみて非常に流暢で整合性のあるテキストを生成でき、要約、翻訳、質問応答など、明示的なタスク専用学習をほとんど行わずとも、高性能なタスク遂行が可能であることが示されました。

**安全性と悪用リスクへの懸念:** OpenAIは、このモデルを全面的に公開することで、偽情報の大量生成やスパムなど悪用リスクを高める可能性を懸念しました。これまでの研究成果では通常、学習に使ったモデルそのものを公開することが多かったのですが、GPT-2に関してはその強力さ故に、最初は完全公開を避け、段階的な公開戦略をとる方針を発表しました。

**「責任ある公開」の実験:** OpenAIは、技術の発展と社会への影響を慎重に考え、新しい公開モデルを模索。研究者コミュニティや社会との対話を行いながら、サイズの小さいバージョンや中規模バージョンを段階的にリリースし、その反応や悪用状況を監視しつつ、最終的にフルモデルを公開するかどうか検討する戦略を取りました。

**研究コミュニティへのインパクト:** 当時、この発表は「高度な自然言語生成モデルをオープンに公開するリスク」と「研究の透明性・再現性を重視する伝統的立場」との対立という新たな局面を象徴するものでした。大規模言語モデルの倫理的・社会的インパクトや、悪用対策への新たな取り組みの必要性を指し示す一例となり、後に他のAI研究機関や企業も、技術公開や大規模モデル開発におけるリスクマネジメントを考慮する流れが強まっていきました。

## Large Language Model

![](https://miro.medium.com/v2/resize:fit:1400/1*na6eVIVet02RemFEjSDA4w.png)

### Scaling Law

Scaling Law（スケーリング則）は、AIモデルの性能が「モデルのサイズ（パラメータ数）」「学習に使用するデータ量」「計算リソース量」の3つの要素を増やすことで向上するという法則です。

![](https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20210103/20210103024804.png)

 [2021-01-05 OpenAIが発見したScaling Lawの秘密](https://deeplearning.hatenablog.com/entry/scaling_law)

モデルのパラメータ数を増やし、より多くのデータで学習させ、強力な計算資源を投入することで、AIの性能が予測可能な形で向上することが示されました。これにより、**研究者やエンジニアは、どの程度リソースを投入すれば目標とする性能を達成できるかを計画しやすくなりました。**

#### 2021年 OpenAI ChatGPT3

![](https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20200720/20200720105817.png)

OpenAIが開発したGPT-3は、スケーリング則に基づき、従来よりもはるかに大規模な1750億個のパラメータを持つモデルとして設計されました。その結果、自然な文章生成や多様なタスクへの対応能力が飛躍的に向上しました。

#### 2022年 OpenAI ChatGPT-3.5

![](https://romptn.com/article/wp-content/uploads/2024/03/スクリーンショット-2024-03-13-122533.png)

### Parameters

パラメータ数を上げていくと小さいモデルでは見られなかったタスクを解けるようになったり，精度が大幅に上昇することが数多くの研究で観測されています．

![](https://aisholar.s3.ap-northeast-1.amazonaws.com/media/September2023/スクリーンショット_2023-09-01_135613.png)

**創発は本当に「質的変化」なのか？**

![](https://aisholar.s3.ap-northeast-1.amazonaws.com/media/September2023/スクリーンショット_2023-09-01_135737.png)

しかし、実際には多くのタスク評価は二値化（正解/不正解）され、性能評価はしばしば「ある閾値を超えたか否か」に強く依存します。そのため、モデルが連続的かつ漸進的に性能を改善していても、スコアの取り方やスケーリングによっては、あたかも不連続なジャンプが起きたように見えることがあります。

定量的かつ精密な分析では、そうした劇的変化の多くが、単なる評価メトリクスの非線形性や、特定タスクでの閾値効果によって説明できるとされています。モデル内部に突然の「新能力」が備わるわけではなく、あくまで性能改善があるメトリクの境界を超えたことが「創発的な飛躍」に見えているに過ぎないというわけです。

- [LLMの「創発」は幻影か](https://ai-scholar.tech/articles/large-language-models/is-emergence-a-mirage)

#### 2023年 OpenAI ChatGPT-4

#### 2023年 Tesla Full Self Driving

![](https://www.thinkautonomous.ai/blog/content/images/2023/09/Screenshot-2023-09-15-at-11.28.19.png)

![](https://www.thinkautonomous.ai/blog/content/images/2023/09/Screenshot-2023-09-15-at-11.29.12.png)

### RLHF

### RLAIF

![](https://i0.wp.com/semianalysis.com/wp-content/uploads/2024/12/150-RLAIF-vs-RLHFGIMP.png?resize=1536%2C675&ssl=1)

### Chain of Thought

![](https://cdn-ak.f.st-hatena.com/images/fotolife/i/izmyon/20230527/20230527054007.png)

従来のLLMは、質問に対して即座に結論を出すことが多く、その背後でどのような推論や思考の飛躍が行われているかは「ブラックボックス」でした。一方、CoTはプロンプト設計の工夫によってモデルに対して「思考プロセスを書き出しながら最終結論に到達するように」促します。このアプローチには以下のメリットがあります。

1. **透明性とデバッグ容易性の向上**：
   モデルがどのような論理的ステップを踏んでいるかがテキストとして明示されるため、ユーザや開発者はモデルの誤りや飛躍を発見しやすくなります。
2. **回答精度の向上**：
   思考過程を明文化することで、モデル自体が推論ミスを減らし、複雑な計算や論理的判断も段階的に行えるようになります。また、途中で必要に応じて前提を再確認することも期待でき、より正確な最終回答が得られやすくなります。
3. **長期的な文脈理解とクリティカルシンキング**：
   CoTは単純なQ&Aより深い問題解決や複雑な質問への対応に適しています。たとえば数学問題や多段階的な論理推論、長い文章の要約、因果関係の明確化など、ステップを踏むことで解決が促される領域で特に有効です。

![](https://cdn-ak.f.st-hatena.com/images/fotolife/i/izmyon/20230527/20230527075006.png)

**研究および事例**
「Chain-of-Thought Prompting Elicits Reasoning in Large Language Models」という研究(Wei et al., 2022)で示されたように、この手法はGPT系モデルや他の大規模言語モデルに対して有効に働きます。論文やブログ記事、IBMなどの技術解説でも、CoTは「モデルが単に確率的に次の単語を予測する」段階から、「論理的思考ステップを内部でシミュレートし、それを明示化する」段階へと踏み出す画期的なプロンプティング手法として位置付けられています。

たとえば複雑な数学問題を解く際、CoTを使わない状態ではモデルは結果を一発で出そうとしてミスが多発する場合があります。しかし、CoTを促すプロンプトを与えることで、「問題の条件整理」→「使用する公式の特定」→「段階的な計算」→「結果確認」という流れをテキストとして書き出し、最終解答の正確性を高めます。

参考文献：

- [What is chain of thoughts (CoT)?](https://www.ibm.com/topics/chain-of-thoughts)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://izmyon.hatenablog.com/entry/2023/05/27/080236)

### Reasoning Scaling

![](https://pbs.twimg.com/media/Gc0zpWDbAAA6T-I?format=jpg&name=medium)

https://x.com/deepseek_ai/status/1859200149844803724

#### 2024年 OpenAI o1

![](https://cdn.openai.com/reasoning-evals/v3/headline-desktop.png?w=3840&q=90&fm=webp)

### AI agent

#### 2024年 Anthropic Compute Use

![](https://techcrunch.com/wp-content/uploads/2024/10/Computer-Use_-Claude-computer-use-demo.png?w=680)

## LLM

### OpenAI GPT

### Google Gemini

### Anthropic Claude

### Meta Llama

### NTT tuzumi

![](https://www.rd.ntt/research/LLM_tsuzumi/fig_03.jpg)

- [NTT版大規模言語モデル「tsuzumi」](https://www.rd.ntt/research/LLM_tsuzumi.html)

<iframe class="speakerdeck-iframe" frameborder="0" src="https://speakerdeck.com/player/838f4169b2854667867bd2da708aa6fb" title="大規模言語モデルとそのソフトウェア開発に向けた応用" allowfullscreen="true" style="border: 0px; background: padding-box rgba(0, 0, 0, 0.1); margin: 0px; padding: 0px; border-radius: 6px; box-shadow: rgba(0, 0, 0, 0.2) 0px 5px 40px; width: 100%; height: auto; aspect-ratio: 560 / 315;" data-ratio="1.7777777777777777"></iframe>
