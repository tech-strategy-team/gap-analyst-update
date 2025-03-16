# Artificial Intelligence

## Introdction

### AIの概念の萌芽

#### 機械が知能を持てるか？

<img src="https://wired.jp/app/uploads/2020/05/19c5f75d150a7e0363d63e1062bba3e5.jpg" style="zoom: 33%;" />

1950年、アラン・チューリングは論文「計算する機械と知能 (Computing Machinery and Intelligence)」を発表し、**「機械が知能を持てるか？」**という問いを立てた。この論文では、後に「チューリングテスト」と呼ばれる基準を提案し、機械が人間のように振る舞えるかどうかを判断する基準を示した。

#### チューリングテスト

チューリングテストは、コンピュータが「人間らしさ」をどれだけ表現できるかを評価するもので、具体的には、審査員がコンピュータと人間の対話者を区別できなければ、そのコンピュータは「知的」と見なされるとされた。

<img src="https://miro.medium.com/v2/resize:fit:758/1*dDv4ExVNwrY-IyaqnmzKcQ.png" style="zoom:50%;" />

人間の審査員が、見えない相手（人間またはコンピュータ）と会話をし、その会話が人間のように自然であれば合格。

### 「人工知能」という言葉の誕生

#### ダートマス会議

<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/z/zawapython/20190226/20190226151651.jpg" style="zoom:50%;" />

人工知能という用語は、この会議で初めて正式に使用された。この名称は、ジョン・マッカーシーが、研究内容を簡潔かつ的確に表現するために提案したもの。当時、計算機科学や認知科学の研究者たちは、機械がどのようにして人間の知能を模倣できるかについて議論を重ねており、この言葉が学際的な研究の旗印として適していると判断された。この会議はジョン・マッカーシー、マービン・ミンスキー、クロード・シャノン、ナサニエル・ロチェスターらによって組織され、ダートマス大学で開催された。会議の目的は、機械が人間のような学習、推論、問題解決などを行える可能性を探ることだった。マッカーシーは、この分野の研究に「人工知能 (Artificial Intelligence)」という名前を付けた。

### AI冬の時代

人工知能研究は1950年代後半から1960年代にかけて大きな期待を集めたが、その後、いくつかの技術的・社会的要因によって進展が停滞する時期を迎えた。この現象は「AI冬の時代」と呼ばれている。

#### 技術的限界

- **シンボリックAIの限界**: 初期のAIは、ルールベースのシステムや論理的推論に依存していましたが、これらの方法は現実世界の曖昧さや膨大なデータに対応出来なかった。例えば、チェスのような特定の問題には効果的でしたが、曖昧な質問や複雑な自然言語に対応することは出来なかった。
- **自然言語処理の課題**: 当時の自然言語処理（NLP）の技術は、主に文法規則や辞書ベースの手法に依存していました。これにより、文の構造を解析することは可能だったが、文脈や多義語の意味を正確に理解することが出来なかった。例えば、"bank"という単語が「銀行」なのか「川岸」なのかを文脈から判断する能力はほとんどなく、翻訳や対話システムでは誤った解釈が頻繁に発生した。また、言語モデルが単純なルールに基づいていたため、文法的に正しいが意味的には不自然な出力が生成されることも多かった。
- **専門システムの制約**: 専門システム（エキスパートシステム）は、特定の分野に関する知識をルールベースでプログラム化し、専門家のように問題を解決するAIシステム。例えば、MYCINは感染症の診断に特化したシステムで、症状や検査結果に基づいて適切な診断や治療法を提案するものだった。しかし、このようなシステムは、ルールが固定的であるため、新しい分野や状況に適応することが難しく、汎用性に乏しいという課題があった。

#### 社会的要因

- **過剰な期待**: 初期のAI研究者たちは、数年以内に人間のように考える機械を実現できると主張しましたが、これらの目標は実現されず、資金提供者の失望を招いた。
- **資金不足**: 実験やプロジェクトが期待通りの成果を上げられなかったため、研究費が削減され、研究活動が縮小した。例えば、アメリカではDARPAがAI研究への資金提供を大幅に減らした。

#### 再興への兆し

- **専門システム**: 1970年代後半には、医療や化学などの専門分野で実用的なAIアプリケーションが開発され、再び注目を集めるようになりました。例えば、MYCINは感染症の診断支援に使用された。
- **日本の第五世代コンピュータ計画**: 1980年代に日本政府が主導したこのプロジェクトは、AI研究の再興を後押ししました。この計画は、並列処理や論理プログラミングを活用した高度なコンピュータの開発を目指しました。
- **検索アルゴリズム**: 情報検索やデータベース技術の向上により、AIは現実的な応用へと活用されるようになりました。例えば、インターネット黎明期には検索エンジンの技術が急速に進化した。
- **プランニングとエージェント**: 自律的に問題解決や意思決定を行うアルゴリズムの研究が進み、ロボットや自律システムに応用され始めました。例えば、ロボットが自動的にタスクを計画して実行する技術が注目を集めた。

このように、AI冬の時代は単なる停滞期ではなく、基礎技術の課題と可能性を見直す重要な期間でもあった。

## Neural Net

<img src="https://miro.medium.com/v2/resize:fit:610/1*SJPacPhP4KDEB1AdhOFy_Q.png" alt="A biological and an artificial neuron" style="zoom:50%;" />

ニューラルネットワークは、人工知能の礎となる技術であり、人間の脳神経系の働きを模倣した計算モデル。その基本的な構造は「ニューロン」と呼ばれる単位が多数連結され、情報を伝達・処理する仕組みから成り立っている。この構造は、複雑なパターンを学習し、未知のデータに対して予測や推論を行う能力を備えている。

- [The differences between Artificial and Biological Neural Networks](https://towardsdatascience.com/the-differences-between-artificial-and-biological-neural-networks-a8b46db828b7)

### Perceptron

<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/k/kakts/20170102/20170102000614.png" style="zoom:50%;" />

**概要**

パーセプトロンは、1958年にフランク・ローゼンブラット（Frank Rosenblatt）によって提案された、単純なニューラルネットワークモデル。このモデルは、入力層と出力層を持ち、入力データに重みをかけた合計値をもとに2値の出力を生成する。数学的には、線形関数を使ってデータを分類するシンプルなアルゴリズムである。

**背景**

1950年代後半、人工知能研究が黎明期を迎える中、機械が「学習」し、パターンを認識できる能力が求められていた。パーセプトロンは、このような期待に応える最初のモデルとして開発された。

**特徴**

パーセプトロンは、簡単なパターン認識や分類タスクを解決することができた。例えば、手書き文字の識別やシンプルな形状の分類といった問題に応用され、従来の固定ルールに基づいたプログラムでは対応できなかった柔軟性を示した。これは機械が自動的に「データから学ぶ」仕組みの可能性を初めて証明したと言える。

**重要性**

パーセプトロンは、機械学習の基本概念である「学習」の実現可能性を初めて示した。特に、簡単なパターン認識タスクにおいて成果を上げ、AI研究への期待を高めた。

**課題**

パーセプトロンには重大な限界があった。それは、線形分離可能なデータしか扱えない点である。たとえば、論理演算の「XOR」問題を解決できないことが、マービン・ミンスキーとシーモア・パパートの著書『パーセプトロン』で指摘され、研究の進展にブレーキをかけた。

### Multi-Layer Perceptron

<img src="https://www.researchgate.net/publication/303875065/figure/fig4/AS:371118507610123@1465492955561/A-hypothetical-example-of-Multilayer-Perceptron-Network.png" style="zoom:50%;" />

**概要**

マルチレイヤーパーセプトロン（MLP）は、ニューラルネットワークの基本形であり、複数の**隠れ層（hidden layers）を持つ全結合型のネットワークである。このモデルは、各層が入力データを処理し、次の層に伝達することで、データの複雑な非線形パターンを学習する能力を持つ。特に、隠れ層と非線形活性化関数**（ReLUやシグモイド関数など）を導入することで、単層パーセプトロンの限界（例: XOR問題）を克服した。

**背景**

MLPは、パーセプトロンの限界を指摘したマービン・ミンスキーとシーモア・パパートの著書『Perceptrons』（1969年）における問題を克服するために開発された。彼らは、単層パーセプトロンでは線形分離不可能な問題（XOR問題）を解決できないことを示した。1970年代から1980年代にかけて、計算能力の向上や理論的ブレイクスルー（例: バックプロパゲーションアルゴリズム）により、MLPが研究されるようになった。

**特徴**

MLPは、以下のような非線形な問題を解決できるようになった

- **XOR問題**: 単層パーセプトロンでは解けなかった線形分離不可能な問題を解決可能。
- **パターン認識**: 手書き文字認識、音声認識、画像分類などの基本的なパターン認識タスク。
- **関数近似**: 非線形関数の近似により、様々な入力と出力の関係を学習可能。

これにより、より複雑なタスクに対してもニューラルネットワークの適用が可能になった。

**課題・限界**

- **計算負荷の高さ**: 隠れ層が増えると計算コストが増大。特に、1980年代には計算資源が十分でなかった。
- **データの不足**: MLPは多くのデータを必要とするが、当時は大規模データセットが限られていた。
- **局所最適解**: 学習時に局所最適解に陥る可能性があり、最適なパフォーマンスを保証できない場合がある。

### Backpropagation

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20240217152156/Frame-13.png" style="zoom:50%;" />

**概要**

バックプロパゲーション（Backpropagation）は、ニューラルネットワークの重みを効率的に学習するためのアルゴリズムである。ネットワークの出力と目標値との間の誤差を計算し、その誤差を逆方向に伝播させることで、各層の重みを更新する。このプロセスは、**勾配降下法（Gradient Descent）**を用いてネットワーク全体の誤差を最小化することを目的としている。

**背景**

バックプロパゲーションは、ニューラルネットワークの学習プロセスを革新した技術だが、その基礎的なアイデアは1970年代にさかのぼる。初期の研究では、ニューラルネットの重み調整が非常に非効率であったため、大規模なネットワークを学習させることが困難であった。

**特徴**

- **多層ニューラルネットワークの学習**: 従来の手法では、隠れ層の重みを効果的に調整できなかったが、バックプロパゲーションにより多層構造が実用化。
- **学習効率の向上**: 勾配降下法を活用することで、誤差の収束が迅速に行えるようになった。
- **非線形問題の解決**: 非線形活性化関数（シグモイド関数など）と組み合わせることで、複雑なデータ構造を学習可能にした。

**課題・限界**

- **勾配消失問題**: 隠れ層が多い場合、誤差が伝播する際に勾配が極端に小さくなる問題（勾配消失）が発生。
- **局所最適解**: 非線形関数を持つネットワークでは、誤差関数が局所的最適解に陥る可能性がある。
- **計算コスト**: ネットワークが大規模になると計算負荷が高くなる。

## Deep Learning

### Convolutional Neural Network

畳み込みニューラルネットワーク

#### 1998 LeNet

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/LeNet-5_architecture.svg/1599px-LeNet-5_architecture.svg.png" alt="LeNet-5 architecture" style="zoom:50%;" />

ルネット（LeNet）は、1990年代初頭にヤン・ルカン（Yann LeCun）によって提案された、畳み込みニューラルネットワーク（Convolutional Neural Network, CNN）の初期モデルである。このモデルは、画像データから局所的な特徴を効率的に学習するために、**畳み込み層**と**プーリング層**を導入した。さらに、これらの特徴を分類するために、全結合層を組み合わせた構造を持っている。特に手書き数字認識（MNISTデータセット）において優れた成果を上げた。

- [Gradient-based learning applied to document recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

#### 2012 AlexNet

<img src="https://miro.medium.com/v2/resize:fit:1400/1*0dsWFuc0pDmcAmHJUh7wqg.png" alt="AlexNet-Architecture" style="zoom:50%;" />

[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

**画像認識技術の課題**

- 画像をうまく分類できなかった
  - コンピューターにとって、画像を認識することは「ピクセルの配列から特徴を見つける作業」である。しかし、従来の方法では手作業で「特徴」を定義する必要があった（例えば、エッジや角の検出など）
  - 画像の数が増えると、計算量が増大し、性能も頭打ちになっていた。
- データが大きくても使いこなせなかった
  - 画像データセット（例えばMNISTやCIFAR-10）はサイズが小さく、大規模なモデルの学習には不十分だった。
  - 高性能なネットワークを学習させるには、膨大な計算が必要だが、当時のCPUでは時間がかかりすぎて実用的ではなかった。

**AlexNetが解決したこと**

- 画像を分類できる＝**深いネットワーク構造（Deep Convolutional Neural Networks）**
  - AlexNetは8層のネットワークを持つモデル
  - **5つの畳み込み層（Convolutional Layers）**：画像から特徴を抽出。
  - **3つの全結合層（Fully Connected Layers）**：抽出した特徴を基に分類を行う。
  - 最後は**Softmax層**を使用し、1000クラスへの分類確率を出力。


この「深さ」が、高度な特徴を学習する鍵となった。

- **GPUを活用した高速学習**

  - AlexNetではNVIDIA GTX 580を2枚使用し、GPUによる並列計算を最適化。

  - ネットワークの一部を各GPUに割り振り、効率的にトレーニングを実施



訓練データのバリエーションを増やすために以下を行った

- ランダムに画像の一部を切り取ったり、左右反転させる。

- RGBチャンネルの輝度をランダムに変更する。

これにより、モデルがさまざまな画像に対応できるようになった。



### Reinforcement Learning (RL)

<iframe width="560" height="315" src="https://www.youtube.com/embed/TmPfTpjtdgg?si=8kN9efSwzSETyRUG" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

ディープラーニングでさまざまなタスクを解けるようになった後、強化学習と組み合わせることで、データが存在しない領域でもAIが学習し、さまざまなタスクを実行可能なことを、DeepMindが示した。

- **強化学習**：エージェントが試行錯誤を通じて最適な行動戦略を学習し、長期的な報酬を最大化することを目指す
- **深層学習**：視覚などの生の入力データから特徴を自動的に抽出し、複雑なパターンを学習する



- [Deep Reinforcement Learning](https://deepmind.google/discover/blog/deep-reinforcement-learning/)
- [西川善司の3DGE：囲碁でトッププロに勝利したDeepMindのAIは，「StarCraft II」でも人間に勝てるか？](https://www.4gamer.net/games/999/G999902/20180404117/)



#### 2016 DeepMinad Alpha Go

DeepMindが囲碁の膨大な手の探索空間を効率的に扱うため、**深層学習（Deep Learning）と強化学習（Reinforcement Learning）**、そして**モンテカルロ木探索（MCTS）**を組み合わせるアプローチを採用し、囲碁でトッププロ棋士を破った。



<img src="https://cdn.packtpub.com/article-hub/articles/40f4e78d2c7769d6840959fc99c50288.png" style="zoom: 50%;" />

- **深層学習**：既存のプロ棋士の棋譜データを使用して、ニューラクネットワークに学習させる。
- **強化学習**：より多様性のあるの打ち方を体験できるように弱いネットワークから強いネットワークまでの打ち手を試して、勝ち負けを判断して、学習させる。



**Policy Network**

- 次の着手（打ち手）の確率分布を予測するネットワークです。
- 盤面を入力すると、その局面で「もっとも有望だと考えられる手」をランキング形式で出力する。

**Value Network**

- 現在の盤面から勝敗の見込みを予測するネットワーク。
- 盤面の状態をスカラー値（勝率など）で評価し、局面の優劣を見積もる。



![](https://cdn-ak.f.st-hatena.com/images/fotolife/s/s7rpn/20160610/20160610190508.png)

**Monte Carlo Tree Search**

- ポリシーネットワークが提示した「有力な手」を軸に、複数の局面をシミュレーションしながら探索する。
- 局面評価にはバリューネットワークを使い、より正確に勝率を見積もる。



- [AlphaGo](https://deepmind.google/research/breakthroughs/alphago/)
- [Mastering the game of Go with deep neural networks and tree search](https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/)
- [Googleが出した囲碁ソフト「AlphaGo」の論文を翻訳して解説してみる。](https://7rpn.hatenablog.com/entry/2016/06/10/192357)



#### 2018 DeepMinad Alpha Zero

![](https://lh3.googleusercontent.com/1CpXd_axBbiiqgZx1hp1F3cume7yA1JO4jG-3PCMiOppkl10G5PcVDRBnhKbhg6s3kmrzbfo_CPFVjOFMsnnsGvLiPl45w0ag5qBHBul3hfxnoCgEk4=w1232-rw)

AplphaGoでは人間の棋譜データから学習し、自己対局による強化学習でトレーニングされているが、AlphaZero は、ゼロから学習して、自己対局のみでチェス、将棋、囲碁などのボードゲームをマスターする。それぞれのゲームで史上最強のプレイヤーとなった。

- チェス： 9 時間
- 将棋：12時間
- 囲碁：１３日



- [AlphaZero: Shedding new light on chess, shogi, and Go](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)



### Transformer

![transformer](https://scrapbox.io/files/676eb7ea5b29ee45de06265a.png)

- [TRANSFORMER EXPLAINER](https://poloclub.github.io/transformer-explainer/)

#### 2017 Google Attention Is All You Need

![](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F61079%2F1eeca783-eee7-8638-1e2f-7f73110bc653.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=4443ae63e855ff709985f643163e4e3b)

Transformerは「自己注意機構（Self-Attention Mechanism）」を使って、すべての単語同士の関係性を一度に捉えられるようになった。これにより、文章の中の重要な単語や文脈を効率的に理解できる。また、**並列処理が可能**なため、学習速度も飛躍的に向上した。Transformerは、入力文章を「エンコーダー」で処理し、意味を理解した後、「デコーダー」で出力文章を生成する。この構造は機械翻訳や要約、質問応答など、幅広いタスクに応用できる。



**従来モデルの課題**

- **RNN（再帰型ニューラルネットワーク）**や**LSTM（長短期記憶）**は、時系列データの処理に優れていたものの、**長い文脈の依存関係**を効果的に捉えることが難しかった。
- 並列計算が困難で、**訓練に時間がかかる**という欠点があった。

**Transformer**

- ##### **自己注意機構（Self-Attention Mechanism）**を用いることで、**入力データ内の全ての単語が相互に関連性を持つ**ようになった。

- これにより、**長い文章内の遠く離れた単語同士の関係性**も効率的に捉えることが可能になった。

- Transformerでは、**Attentionメカニズム**により全てのトークンを**並列処理**できるようになった。

- Transformerのアーキテクチャは、大規模なテキストデータを用いた**事前学習（Pre-training）**と、特定のタスク向けの**ファインチューニング（Fine-tuning）**の組み合わせに適していた。

- **事前学習**で広範な知識を獲得し、**ファインチューニング**で特定のタスクに最適化することで、自然言語理解や生成のタスク性能が飛躍的に向上した。



- [Attention is All You Need](https://research.google/pubs/attention-is-all-you-need/)
- [論文解説 Attention Is All You Need (Transformer)](https://deeplearning.hatenablog.com/entry/transformer)



#### 2018 OpenAI GPT-1

<img src="https://scrapbox.io/files/676ea00d4ff4e4615eb238ef.png" style="zoom: 33%;" />

GPT-1は、2017年に発表されたGoogleの論文「Attention Is All You Need」によって提唱された**Transformerアーキテクチャ**に基づいて構築された。従来のRNN（再帰型ニューラルネットワーク）やLSTM（長短期記憶ネットワーク）よりも並列処理が容易で、大規模データセットでの学習効率が高いことが特徴。

GPT-1は**「タスク固有モデル」から「汎用モデル」**への転換を示した。従来はタスクごとに専用モデルを設計・訓練していたのに対し、GPT-1は1つの事前学習済みモデルをタスクごとに微調整することで幅広い言語タスクに対応できることを実証した。

- [Improving language understanding with unsupervised learning]()
- [Github finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)



#### 2019 OpenAI GPT-2

<img src="/Users/kouichihara/Library/Application Support/typora-user-images/スクリーンショット 2024-12-08 16.48.46.png" alt="スクリーンショット 2024-12-08 16.48.46" style="zoom: 33%;" />

[Better language models and their implications](https://openai.com/index/better-language-models/)

**モデルの大規模化と性能向上:** GPT-2は、事前学習モデルの一種で、従来よりも遥かに大きなモデル（15億パラメータ）と幅広いデータを用いたことで、高度な言語理解・生成性能を示しました。

**驚くべき生成能力:** 当時の水準からみて非常に流暢で整合性のあるテキストを生成でき、要約、翻訳、質問応答など、明示的なタスク専用学習をほとんど行わずとも、高性能なタスク遂行が可能であることが示されました。

**安全性と悪用リスクへの懸念:** OpenAIは、このモデルを全面的に公開することで、偽情報の大量生成やスパムなど悪用リスクを高める可能性を懸念しました。これまでの研究成果では通常、学習に使ったモデルそのものを公開することが多かったのですが、GPT-2に関してはその強力さ故に、最初は完全公開を避け、段階的な公開戦略をとる方針を発表しました。

**「責任ある公開」の実験:** OpenAIは、技術の発展と社会への影響を慎重に考え、新しい公開モデルを模索。研究者コミュニティや社会との対話を行いながら、サイズの小さいバージョンや中規模バージョンを段階的にリリースし、その反応や悪用状況を監視しつつ、最終的にフルモデルを公開するかどうか検討する戦略を取りました。

**研究コミュニティへのインパクト:** 当時、この発表は「高度な自然言語生成モデルをオープンに公開するリスク」と「研究の透明性・再現性を重視する伝統的立場」との対立という新たな局面を象徴するものでした。大規模言語モデルの倫理的・社会的インパクトや、悪用対策への新たな取り組みの必要性を指し示す一例となり、後に他のAI研究機関や企業も、技術公開や大規模モデル開発におけるリスクマネジメントを考慮する流れが強まっていきました。



## Large Language Model

![LLM traing](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4d4ee2d-c249-4b5f-8dae-a82cd648e990_1600x549.png)



- [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training)



### Preprocessing (Tokenization etc...)

![Tokenization](https://miro.medium.com/v2/resize:fit:1200/format:webp/1*erWXUYwg1qlNekZbLK_ozg.png)

Tokenizer とは、入力されたテキストデータを “トークン” という小さな単位に分割する仕組み・アルゴリズムのことを指す。
OpenAIのText Generationモデル（GPT-4など）やEmbeddingsモデルは、テキストを**トークン**単位で処理する。具体的には、英語などでは「単語の先頭の空白も含む」「よく登場する文字列は1トークンとしてまとめる」といった独自のルールに基づきテキストを切り出していく。大規模言語モデルは、文章を直接一つの塊で処理するのではなく、より細かい “トークン” 単位で解析と生成を行う。こうすることで、複雑な文脈を高い精度で理解し、次に続く単語を予測できるようになる。

- [OpenAI tokenizer](https://platform.openai.com/tokenizer)



<img src="https://scrapbox.io/files/676e025e72818e3f96fda602.png" style="zoom:50%;" />



### Pre-training

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*p_vXgNorVabMaiiT.png)

LLM は膨大なコーパスから、言語の統計的パターンを “自己教師あり学習” (self-supervised learning) で学習する。具体的には、自己完結的な予測タスク（例: マスクしたトークンを当てる、次の単語を予測する）などを通じて言語表現を獲得する。モデルはテキストの文脈把握や文法構造など、言語的な基礎能力を獲得する。ただし、この段階ではまだユーザに最適化された回答生成やタスク特化型の能力は不十分。

- [Fine-Tuning LLMs ( Large Language Models )](https://blog.stackademic.com/fine-tuning-llm-large-language-models-b61eb2be2275)



### Scaling Law

Scaling Law（スケーリング則）は、AIモデルの性能が「モデルのサイズ（パラメータ数）」「学習に使用するデータ量」「計算リソース量」の3つの要素を増やすことで向上するという法則。

<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20210103/20210103024804.png" style="zoom:50%;" />

 [2021-01-05 OpenAIが発見したScaling Lawの秘密](https://deeplearning.hatenablog.com/entry/scaling_law)

モデルのパラメータ数を増やし、より多くのデータで学習させ、強力な計算資源を投入することで、AIの性能が予測可能な形で向上することが示された。これにより、**研究者やエンジニアは、どの程度リソースを投入すれば目標とする性能を達成できるかを計画しやすくなった。**



#### 2021 OpenAI GPT-3

<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20200720/20200720105817.png" style="zoom:67%;" />

OpenAIが開発したGPT-3は、スケーリング則に基づき、従来よりもはるかに大規模な1750億個のパラメータを持つモデルとして設計された。その結果、自然な文章生成や多様なタスクへの対応能力が飛躍的に向上した。



<img src="https://storage.googleapis.com/zenn-user-upload/iyxpaqr82a5c8yuopd2rta5dov5z" style="zoom:50%;" />



- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [【論文】Language Models are Few-Shot Learners](https://zenn.dev/dhirooka/articles/dc3d31f15cccb6)



### Reinforcement Learning From Human feedback (RLHF)⁠

<img src="https://images.ctfassets.net/kftzwdyauwt9/12CHOYcRkqSuwzxRp46fZD/928a06fd1dae351a8edcf6c82fbda72e/Methods_Diagram_light_mode.jpg?w=2048&q=80&fm=webp" style="zoom: 33%;" />

OpenAIのInstructGPT モデルをトレーニングするための中核技術は、 [人間のフィードバックからの強化学習 (RLHF) ⁠](https://openai.com/index/deep-reinforcement-learning-from-human-preferences/)。これは、以前のアライメント研究で先駆者となった方法である。この技術では、人間の好みを報酬信号として使用してモデルを微調整する。

- [OpenAI Aligning language models to follow instructions](https://openai.com/index/instruction-following/)



#### 2022 OpenAI Instruct-GPT

![](https://scrapbox.io/files/676f50b51888c1ee7e0d9dec.png)

従来のGPT-3モデルは、大規模なインターネットテキストで訓練されているため、ユーザーの意図に完全には沿わない出力を生成することがあった。これに対し、InstructGPTは人間のフィードバックを活用した強化学習（RLHF）を用いることで、ユーザーの指示により正確に応答するよう調整されている。

**InstructGPTの訓練プロセス:**

1. **教師ありファインチューニング（SFT）:** 人間のラベラーが提供する望ましい出力例を使用して、モデルをファインチューニングする。これにより、モデルは人間の回答パターンを学習する。
2. **報酬モデル（RM）の訓練:** ラベラーがモデルの出力を評価し、好ましい順にランク付けする。このデータを基に、モデルが高品質な出力を予測できるよう報酬モデルを訓練する。
3. **強化学習（PPO）:** 報酬モデルを用いて、モデルの出力がユーザーの指示に適切に従うよう最適化する。これにより、モデルの性能が継続的に向上する。

これらのステップを反復的に行うことで、InstructGPTはユーザーの意図により忠実に応答できるようになる。実際、1.3億パラメータのInstructGPTモデルは、1750億パラメータのGPT-3モデルよりもユーザーの指示に従う能力で高く評価されている。



- [Aligning language models to follow instructions](https://openai.com/index/instruction-following/)
- [自然言語処理技術の進化：AI による「ことば」の処理から汎用 AI へ 最近の動向について](https://www.ipa.go.jp/digital/chousa/trend/ai-technologies/f55m8k0000007bdz-att/000098829.pdf)



#### 2022 OpenAI ChatGPT

![ChatGPT](https://www.oca.ac.jp/itmagazine/wp-content/uploads/2022/12/ChatGPT02-1024x461.png)

OpenAIはGPT-3.5をベースに、チャット形式での会話するためのモデルとしてトレーニングしたChatGPTを作成した。GPT-3と比較して、GPT-3.5は「指示に従う能力（Instruction Following）」が飛躍的に向上し、曖昧な質問や複雑なリクエストにも柔軟に応答できるようになった。人間によるフィードバックからの強化学習（RLHF）によって、チャット形式の会話ができるようになっている。



**ChatGPTの制約**

- ChatGPT は、もっともらしく聞こえるが、正しくなかったり意味不明な回答をすることがある。

- 入力フレーズの微調整や、同じプロンプトを複数回試行することに敏感

  例）質問のフレーズが 1 つ与えられた場合、モデルは答えを知らないと主張するが、少し言い換えると、正しく答える



- [OpenAI Introducing ChatGPT](https://openai.com/index/chatgpt/)
- [ChatGPTはどのように学習を行なっているのか](https://zenn.dev/ttya16/articles/chatgpt20221205)
- [ChatGPT 人間のフィードバックから強化学習した対話AI](https://www.slideshare.net/slideshow/chatgpt-254863623/254863623)



### Parameters

パラメータ数を上げていくと小さいモデルでは見られなかったタスクを解けるようになったり，精度が大幅に上昇することが数多くの研究で観測されている。

<img src="https://aisholar.s3.ap-northeast-1.amazonaws.com/media/September2023/スクリーンショット_2023-09-01_135613.png" style="zoom: 50%;" />

**創発は本当に「質的変化」なのか？**

<img src="https://aisholar.s3.ap-northeast-1.amazonaws.com/media/September2023/スクリーンショット_2023-09-01_135737.png" style="zoom:50%;" />

しかし、実際には多くのタスク評価は二値化（正解/不正解）され、性能評価はしばしば「ある閾値を超えたか否か」に強く依存する。そのため、モデルが連続的かつ漸進的に性能を改善していても、スコアの取り方やスケーリングによっては、あたかも不連続なジャンプが起きたように見えることがある。

定量的かつ精密な分析では、そうした劇的変化の多くが、単なる評価メトリクスの非線形性や、特定タスクでの閾値効果によって説明できるとされている。つまり、モデル内部に突然の「新能力」が備わるわけではなく、あくまで性能改善があるメトリクの境界を超えたことが「創発的な飛躍」に見えているに過ぎないという。



- [LLMの「創発」は幻影か](https://ai-scholar.tech/articles/large-language-models/is-emergence-a-mirage)



### Mixture-of-Experts

![Mixture-of-Experts](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/mixtral-8x7b-model-possible-interpretation-diagram.png)

従来、モデルの性能向上にはパラメータ数の増加が有効とされてきたが、計算コストや学習時間の増大が課題となっていた。MoEは、必要に応じて一部のエキスパートのみを活性化する「スパース」な構造を持つことで、計算リソースを節約しながらモデルの容量を拡大するアプローチとして注目されている。



- [The AI Brick Wall – A Practical Limit For Scaling Dense Transformer Models, and How GPT 4 Will Break Past It](https://semianalysis.com/2023/01/24/the-ai-brick-wall-a-practical-limit/)
- [Applying Mixture of Experts in LLM Architectures](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/)
- [【論文瞬読】大規模言語モデルの新潮流：Mixture of Experts (MoE) の可能性と課題](https://note.com/ainest/n/n997d0d8ecb90)



#### 2023 OpenAI ChatGPT-4

<img src="https://cdn.buttercms.com/output=f:webp/dnmV4PvuQ42A8fQufWiz" style="zoom: 50%;" />

GPT-4はOpenAIが開発した大規模マルチモーダルAIモデルで、テキストと画像を入力として受け取り、テキストを出力できる。GPT-4はGPT-3.5より複雑なタスクで信頼性や創造性が向上し、模擬司法試験で上位10％の成績を達成した。トレーニングにはAzureと共同設計したスーパーコンピューターが使用され、安全性や正確性の向上に注力されている。ただし、依然として事実誤認やバイアスの問題は完全には解消されていない。

<img src="https://scrapbox.io/files/676d5e2990765fbc66806017.png]" alt="Exam results GPT-4" style="zoom:33%;" />



- [OpenAI GPT-4](https://openai.com/index/gpt-4-research/)
- [GPT-4 Technical Report](https://arxiv.org/html/2303.08774v6)
- [GPT-4 Architecture, Infrastructure, Training Dataset, Costs, Vision, MoE ](https://semianalysis.com/2023/07/10/gpt-4-architecture-infrastructure/)



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
「Chain-of-Thought Prompting Elicits Reasoning in Large Language Models」という研究で示されたように、この手法はGPT系モデルや他の大規模言語モデルに対して有効に働きます。論文やブログ記事、IBMなどの技術解説でも、CoTは「モデルが単に確率的に次の単語を予測する」段階から、「論理的思考ステップを内部でシミュレートし、それを明示化する」段階へと踏み出す画期的なプロンプティング手法として位置付けられています。

たとえば複雑な数学問題を解く際、CoTを使わない状態ではモデルは結果を一発で出そうとしてミスが多発する場合があります。しかし、CoTを促すプロンプトを与えることで、「問題の条件整理」→「使用する公式の特定」→「段階的な計算」→「結果確認」という流れをテキストとして書き出し、最終解答の正確性を高めます。



- [What is chain of thoughts (CoT)?](https://www.ibm.com/topics/chain-of-thoughts)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://izmyon.hatenablog.com/entry/2023/05/27/080236)



### Reinforcement Learning with AI Feedback (RLAIF) 

<img src="https://i0.wp.com/semianalysis.com/wp-content/uploads/2024/12/150-RLAIF-vs-RLHFGIMP.png?resize=1536%2C675&ssl=1" style="zoom:50%;" />

**RLAIF（AIフィードバックによる強化学習）** は、人間のフィードバックを用いる従来の**RLHF（Reinforcement Learning from Human Feedback）** に代わる、よりコスト効率が良く拡張性に優れた手法です。
この手法では、**人間ではなく大規模言語モデル（LLM）**がフィードバックを生成し、そのフィードバックをもとに**報酬モデル（Reward Model, RM）**を訓練します。



**RLHFとの比較**

- **RLHF:** 人間のフィードバック（例：どちらの回答が良いかなど）を用いて報酬モデルを訓練し、その後、強化学習を行う。
- **RLAIF:** LLMがフィードバックを生成し、そのフィードバックを基に報酬モデルを訓練し、強化学習を行う。

**RLAIFの利点**

1. **コスト効率:** 人間のアノテーションに比べて、LLMを使用したフィードバック生成は**10倍以上安価**。
2. **拡張性:** 高品質なフィードバックを大規模に生成可能。
3. **柔軟性:** Direct-RLAIFにより、報酬モデルの訓練工程をスキップできる。
4. **性能:** RLAIFはRLHFと同等かそれ以上の性能を示す。



- [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267)

- [Alignment faking in large language models](https://www.anthropic.com/news/alignment-faking)



### Synthetic data

<img src="https://assets.st-note.com/img/1716371922613-v155uhi5y6.png?width=1200" style="zoom:50%;" />

LLMを用いた合成データ生成には主に以下の2つの手法がある

1. **蒸留（Distillation）**: より大規模で高性能なLLMを使用して、合成データを生成する方法。この手法により、人間が作成したデータに近い高品質な合成データを作成し、例えば7B（70億）のパラメータを持つモデルの性能を向上させることが可能。
2. **自己改善（Self-improvement）**: モデル自身がデータセットを生成し、そのデータで自己強化を行う方法。特に、より大きなサイズのモデルが存在しない場合やライセンスの制約がある場合に有効で、例えば70B（700億）のパラメータを持つモデルの学習に利用される。

- [LLMによる合成データ(Synthetic Data)生成のテクニック](https://note.com/hatti8/n/n193430331561)
- [Using LLMs for Synthetic Data Generation: The Definitive Guide](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms)
- [Best Practices and Lessons Learned on Synthetic Data for Language Models](https://arxiv.org/html/2404.07503v1)



### Reasoning Scaling



<img src="https://pbs.twimg.com/media/Gc0zpWDbAAA6T-I?format=jpg&name=medium" style="zoom: 50%;" />

CoTにより思考プロセスを出力させることで、回答精度が上がることが判明し、それを元に発展されたアプローチとして、推論スケーリングというアイデアが登場した。



- 強化学習を用いて、正しい推論パス（CoT）を理解させる
- 推論時間をスケールアップさせる



- [Deepseek-r1](https://x.com/deepseek_ai/status/1859200149844803724)
- [Improving LLM Reasoning through Scaling Inference Computation with Collaborative Verification](https://arxiv.org/html/2410.05318v1)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/pdf/2407.21787)
- [Scaling Laws – O1 Pro Architecture, Reasoning Training Infrastructure, Orion and Claude 3.5 Opus “Failures”](https://semianalysis.com/2024/12/11/scaling-laws-o1-pro-architecture-reasoning-training-infrastructure-orion-and-claude-3-5-opus-failures/)





#### 2022 Google STaR

![](https://scrapbox.io/files/6776334e2057272d1af302f9.png)

大規模言語モデルのトレーニングは膨大なデータセットが必要となる。CoTを用いることで精度が向上することが研究から判明しているため、推論能力を向上させる方法として、ヒントを参考にLLMが問題を解いて、そのQAデータを大規模に生成し、学習データとするアプローチが考え出された。AIがAIによる強化学習によって自身をブートストラップするループである。



- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)



#### 2024 Google Quiet-STaR 

![](https://scrapbox.io/files/6776355c5bd979ec3fdc6376.png)

STaRではヒントをベースに推論を行うアプローチのため、ヒントが制約条件となっていた。Quiet-STaRではCoTの推論プロセス自体を強化学習によって、学習することで汎用的かつスケーラブルな推論方法を獲得する。

- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)





#### 2024 OpenAI o1

<img src="https://cdn.openai.com/reasoning-evals/v3/headline-desktop.png?w=3840&q=90&fm=webp" style="zoom: 33%;" />

OpenAIは、複雑な問題を解決するための新しいAIモデル「o1シリーズ」を発表しました。このシリーズは、より多くの時間をかけて「考える」ことができ、科学、コーディング、数学などの分野で難解なタスクに対して優れた推論能力を発揮します。特に、物理、化学、生物学などのチャレンジングなベンチマークテストで、博士課程レベルの学生に匹敵する結果を出し、国際数学オリンピック（IMO）では従来のモデルと比べてはるかに高い精度で問題を解決しています。安全性についても強化されており、新しいモデルは「安全ルール」を守る能力も向上しており、ユーザーがルールを回避しようとする試みにも高い耐性を示しています。



- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)



#### 2024 OpenAI o3

<img src="https://media.datacamp.com/cms/ad_4nxfcwblnnivdsqerowrleinsvqk2k2mpbxfto8mrd2uqxliykoak8n_g1qthbuh0e4tuugaoyqjfdkjiyh4ntfvrlewzq_42ftdnqv-oup5h4prxsm_5jwi7-jmqhx2ly1cpf2253g.png" style="zoom:50%;" />

o3シリーズは、複雑な問題に対する推論能力を強化することを目的としている。特に、コーディングや高度な数学、科学の分野で優れた性能を示している。例えば、o3は高度な数学コンペティションであるAIME 2024で96.7%の正答率を達成し、前モデルのo1の83.3%を大きく上回った。また、科学分野の専門的な質問に対する精度を測るGPQA Diamondベンチマークでは、87.7%の精度を記録している。さらに、ソフトウェアエンジニアリングのベンチマークであるSWE-Bench Verifiedでは、o3は71.7%の精度を達成し、o1の48.9%から大幅に向上している。これらの成果は、o3が複雑なタスクにおける推論能力を大幅に向上させたことを示している。

- [OpenAI o3 and o3-mini—12 Days of OpenAI: Day 12](https://www.youtube.com/live/SKBG1sqdyIU?si=0nMc0kWc_xBfDBiZ)
- [OpenAI’s O3: Features, O1 Comparison, Release Date & More](https://www.datacamp.com/blog/o3-openai)

**o3 mini**

![o3 mini](https://media.datacamp.com/cms/ad_4nxftfojc_9ihmypaa8pe9fjxtdc7rdffhr98vjz9e5socbu_owoz3yd9irpq94qupviojbtvmolzkaap8_0hdffftex8ngvlfik-ohkh97h_7a0wliecrrgktn9jrrradcuvz-h6.png)

o3は高い能力を示すが、それと同時に高い計算コストを要求するため、実用的なコストに収まる小型モデルとしてo3-miniも同時に発表された。コストパフォーマンスにおいてはo1-miniに近いレンジに収まっており、o3の1/10程度のコストとなっている。



##### AGI benchmark

<img src="https://arcprize.org/media/images/arc-example-task.jpg" style="zoom:50%;" />

**ARC-AGI**はAI における最も困難な未解決の問題に注目するように設計された研究ツールである。このベンチマークは、**AIが未知のタスクに対して新しいスキルを効率的に習得できる能力**、すなわち一般化能力を測定することを目的としている。

**概要**

ARC-AGIは、さまざまな入力と出力の例から構成されるタスクセットで、各タスクは色付きのグリッド形式で表現される。AIシステムは、提供された例に基づいてパターンを認識し、**未見のテスト入力に対して正しい出力を生成することが求められる。**このプロセスは、人間が新しい問題に直面したときに推論や抽象化を行う方法に似ている。

**設計**

ARC-AGIの設計には、人間が自然に持つ基本的な知識（オブジェクトの存在、目的志向性、数と計数、基本的な幾何学とトポロジー）に基づく「コア知識プライア」が組み込まれている。これにより、AIと人間の知能を公平に比較することが可能となる。

**結果**

<img src="https://aisouken.blob.core.windows.net/article/20000/Arc-AGI%20Score.webp" style="zoom: 33%;" />

GPT-3のスコアは0、GPT-4のスコアは5%で、今までの手法では突破が困難だった。しかし、o1、o3と推論スケーリングを採用したアプローチでは急速にスコアを伸ばし、今までのLLMとは異なり、**o1/o3は学習していない未知の問題に対しても対応ができるようになったという質的な変化が起きている**ことがわかる。

**コスト**

<img src="https://arcprize.org/media/images/blog/o-series-performance.jpg" style="zoom:50%;" />

しかし、o1/o3シリーズは新しい課題も生み出している。今までのLLMは学習フェイズで大量の計算資源を必要とするが、推論時には比較的低コストで計算できるため、ビジネスとして大きな投資に見合う成果が得られるアーキテクチャだった。推論スケーリングは学習時だけでなく、**推論時にも計算資源を必要とするため、提供コストもスケールしてしまい、ビジネス的なスケールメリットが発生しないアーキテクチャとなっている**。今回のベンチマークでは1000ドルもかけて問題を解いているため、この解決のために、**今後は推論に特化した半導体ニーズが高まる。**



- [Defining AGI](https://arcprize.org/arc)
- [OpenAI o3 Breakthrough High Score on ARC-AGI-Pub](https://arcprize.org/blog/oai-o3-pub-breakthrough)



## Large Concept Model

<img src="https://scrapbox.io/files/676ac8546613ec1cd3f90b12.png" alt="Large concept model" style="zoom: 67%;" />

LLM（大規模言語モデル）は人工知能分野に革命をもたらし、多くのタスクで標準ツールとして使用されているが、これらは主にトークンレベルで動作している。本研究では、「コンセプト」と呼ばれる高レベルの意味表現に基づく新しいアーキテクチャ「Large Concept Model (LCM)」を提案する。LCMは言語やモダリティに依存せず、SONARと呼ばれる文埋め込み空間を利用して自動回帰型の文予測を行う。1.6Bパラメータのモデルで複数のアプローチを評価し、最終的には7Bパラメータモデルに拡張して、要約や新タスク「要約拡張」で優れたゼロショット性能を示した。本モデルは200以上の言語をサポートし、コードは公開されている。

- [Large Concept Models: Language Modeling in a Sentence Representation Space](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)





## What's next



### Pre traning End

<iframe width="560" height="315" src="https://www.youtube.com/embed/1yvBqasHLZs?si=cieEG5XZ44uzl0U7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

> しかし、事前学習には限界があります。それは「データは有限である」という事実です。私たちには一つのインターネットしかなく、データの成長には限界があります。この制約を超えるために「合成データ」や「エージェント型AI」などのアプローチが模索されています。
>
> Ilya Sutskever



### Synthetic Data

<img src="https://pbs.twimg.com/media/GY_Or3GXQAAxtPQ?format=jpg&name=large" style="zoom:50%;" />

> キャンバスモデルはすべて合成データでトレーニングされており、私たちのアプローチと、それがどのようにして2か月間でコア動作を非常に迅速にモデルにトレーニングすることを可能にしてきたかについてもう少し詳しく説明しました。
>
> OpenAI Karina Nguyen
>
> https://x.com/karinanguyen_/status/1841918534013571139





### Reinforcement Learning 

<img src="https://pbs.twimg.com/media/GfWzffNaYAAki4c?format=jpg&name=medium" style="zoom:50%;" />

> Gemini 2.0 Flash。なんで軽量モデルがこの性能を出せると言うと、最大大規模モデルを持ってるから。現在、バカでかい調達をした基盤モデル系の大部分のスタートアップが頓挫してるのも、ようやく投資家が気づき始めたからです。日本も注意しましょう。0から最高の軽量モデルは生まれないです。
>
> Google Deep Mind Shane Gu
>
> https://x.com/shanegJP/status/1866893458247651518



- [Scaling Laws – O1 Pro Architecture, Reasoning Training Infrastructure, Orion and Claude 3.5 Opus “Failures” ](https://semianalysis.com/2024/12/11/scaling-laws-o1-pro-architecture-reasoning-training-infrastructure-orion-and-claude-3-5-opus-failures/)



## Future

- [OpenAI Sam Altman The Intelligence Age](https://ia.samaltman.com/)
- [Anthropic Machines of Loving Grace](https://darioamodei.com/machines-of-loving-grace#taking-stock)
- [Google DeepMind A new golden age of discovery](https://deepmind.google/public-policy/ai-for-science/)

