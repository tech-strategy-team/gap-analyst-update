## 大規模言語モデル

### 大規模言語モデルを使ってみる

「DXを進めるにはどうすればいい？」

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhJ5XrlGZKwN3Q_hEk139JOvb3Ieg5bC08jOqftLpESRRQ6_v4appLaa55PGR4g_1eK3A73UBrF_PaA8XsfswRgPPShCgZRkG8yHMvEIJNllUq3g14Pok0UGjtNZRVl3PNrLcbLxSfLX7TZ/s550/ai_shigoto_makaseru.png" style="zoom: 50%;" />

```
デジタルトランスフォーメーション（DX）を進めるには、以下のステップを考慮することが重要です。
それぞれの段階で組織や状況に応じたアプローチを取ることで、効率的かつ効果的なDXの実現が可能です。

1. ビジョンと目標の設定
 • 目的を明確化: DXを通じて何を達成したいのか、組織全体で共通認識を持つ。
 • 例: 業務効率化、新しい収益源の確立、顧客体験の向上。
 • KPIの設定: 成果を測定する指標を定め、進捗を確認。
(中略)
```

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjs6pYvoioYkbu0iulzyVynH1_Ep287KdAdFGoGATHz-fmJm3TOef703ZAhEom8GD1X3stxjkGujLnLq5E0VPQxTLvER9_r6YpvkbkPnxPNR_sqc1nP8hXt12mpNjBshG73TuyVK4lukWCR/s800/face_angry_man3.png" style="zoom:33%;" />

DXの基本定義や一般論（「顧客体験を改善する」「業務効率化を図る」など）が返され、具体性がなく、「それくらい知ってる」と感じる。

### なぜ抽象的な回答になりやすいのか？

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg2dvuvi_VSVGkzORLeRYSzJid_WnW87vr8evE0R3UsfUvf-Mp5A9Ck5A6z5Nf8MwklomuG4s8lBtqE0EMieboxTAhyphenhyphenXzN8OGY_DfnFhsA_adweiu-CyYNWHRPrEf9rQ2C2-FAl6h2cTJfB/s400/ai_study_kikaigakusyu.png" style="zoom: 50%;" />

質問が曖昧で「前提条件が少ない」と、 AIが「どのような具体例や前提が適切なのか」を判断しづらくなります。例えば、「どうやったら上手く生きられますか？」という質問はとても広くて、人によって答えは全然違います。モデルには、特定の人間と状況(年齢、職業、場所)などの情報がないため、何か共通して言えそうな、当たり障りのない「一般論的」な答えを出しやすくなります。

- AIは「具体的状況」や「明確なゴール」がないと、文章データからよくある「無難な」パターンを引き出してしまう。

- この「無難な」パターンは、大抵「こんなことに気を付けましょう」「目標を立てて行動しましょう」などの抽象的な表現になりがち。

- つまり、 AIは本質的に「意味を理解」しているわけではなく、「よくある回答パターン」を確率で選んでいるため、条件がはっきりしないと「みんながよく使う抽象的なアドバイス」程度の答えになってしまいます。

#### 基本的な仕組み

![スクリーンショット 2024-12-08 15.49.49](/Users/kouichihara/Library/Application Support/typora-user-images/スクリーンショット 2024-12-08 15.49.49.png)

**前提1:** 大規模言語モデルは、大量のテキストを学習しており、そのテキスト中で「次に来そうな言葉」を予測して文章を作っています。

**前提2:** 大規模言語モデルは特定の知識やルールを「辞書」のように入れているわけではなく、たくさんの文章パターンをもとに、「こういう文脈ならこんな答えがよく使われる」という確率的なパターンを使って答えを組み立てています。

#### 回答するときにやっていること

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEicb1iF4KAWF_pl99nrSTJxMt1Kso1X6Va9jADFVFA2lBmmpfQYeAVCMa1L5XeFB0TO7pE0P1gW6WAo6dZ5OUAo4ONcu1wq3T-FRc1TPQWsq_3PyKUtLMMPAP1m4qF4_HS2hoxjUx8v6DNR/s400/ai_shigoto.png" style="zoom: 50%;" />

- 質問文を入力すると、モデルは各単語に「どれくらい重要か」を点数（重み）として計算する。

- 曖昧な質問では、どの単語に重みを割り振れば良いかわからず、すべての単語が均等に扱われる傾向。

- 結果的に、生成される単語の確率分布が「均等」に近づく。

**曖昧な質問：「どうやったら上手く生きられますか？」**

```
単語候補とその確率：
 • 「努力」：20%
 • 「挑戦」：18%
 • 「好きなこと」：15%
 • 「幸せ」：12%
 • 「計画」：10%
 • …（多くの選択肢が似た確率で並ぶ）
```

**具体的な質問：「中学生が部活と勉強を両立するには？」**

```
単語候補とその確率：
 • 「時間管理」：50%
 • 「スケジュール」：30%
 • 「休憩時間の活用」：15%
 • 「部活の優先順位」：5%
```

#### AIは人間と同じ条件で回答していない

![](https://assets.st-note.com/img/1731490231-QZ2PCA9FbgUyxzSjoWcr1RVJ.png?width=1200)

あなたが先生に「将来成功するにはどうすればいいですか？」と聞くと、先生はあなたの興味、得意なこと、目標、置かれている環境を知らなければ、「努力しようね」「目標を持とうね」といった抽象的なアドバイスしかできません。それと同じで、LLMにも前提や具体的な状況が無いと、当たり障りのない抽象的なアドバイスを繰り返してしまうのです。

[「AIオンボーディング」の重要性とAi Workforceの挑戦](https://note.com/nrryuya/n/nc03ba3c806e3)

### プロンプトの基本

#### 明確で具体的に指示を出す

#### 文脈を提供する

#### フォーマットを指定する

### プロンプト特有のテクニック

### 具体的な使い方

#### 情報の変換

##### 要約する

##### フォーマットに合わせて出力する

#### ドキュメントのQA

##### 仕様書を読む

##### 論文を読む

##### ソースコードを読む

#### ドキュメントレビュー

##### 説明資料をレビューさせる

##### 感想文をレビューさせる

#### 簡易なプログラミングで解決させる

##### 文字数を数えさせる

##### 特定の単語の出現数を数えさせる

##### テキストの比較

##### ファイル名の変更

#### 技術的な課題解決に使う

##### 特定の技術を解説してもらう

##### 特定の技術の使い方を教えてもらう

##### エラーのデバッグをする

##### 問題の解決策について相談する

##### 問題について簡単な試算をする

#### エクセルを操作する

##### データの抽出

##### グラフを作成する

##### データの解釈をしてもらう

#### 画像を入力として使う

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhzbRYLmbUY4N5zDeVOYP68i5q7-3eYUCn7lTy99lbJPNtq0ORlrBmT_klgkJwhfHpi2h9OKVv6Ewqy70nDzLv5G4bTP3PzMLuQNg2xMp8k-ks5iC-AYc2ApFgrVy2QUtjQh7QyH56fe4k/s400/ai_image_gazou_ninshiki.png" style="zoom:50%;" />

##### 紙の情報ををテキスト化する

##### 紙に書かれた情報をエクセルにする

#### 教師として使う

##### マネジメントの悩みを相談する

##### プログラミングを教えてもらう

##### トレーニング計画を考えてもらう

#### 歌の歌詞を書く

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiKBimAcrO1MWKThSWtn8XEuerJXEQU-ZBZVeteLwth7mPvjkTMDHFaX_3u9POEjZAmtcQElNmatRumR14_9ezabibWOnWMkYkIaHRQA6Akjk50Z0rgHSN7_6Wtx8-7e18si2JYhFdS_tHp/s450/music_kashi_card.png" style="zoom:50%;" />

### LLMを使ったアプリケーション

#### ChatUIはLLMのゴールか？
