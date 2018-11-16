VAEを使った画像生成
=
入力した画像をオートエンコーダモデルで画像の生成を行う

データセット
-
画像検索エンジンで検索した結果の画像をデータセットの基とした。

その画像に対して、opencvの顔検出を用いて顔にフォーカス、(256, 256)にリサイズしたRGB画像を生成する。これをデータセットとした。

モデル構成
-
入力と出力は256×256の画像のRGB(3チャネル)

VAE
--
encoderは3×256×256の画像Tensorを受け取り、ベクトルmu, varを作る。

mu, varを平均、標準偏差とするガウス分布を仮定し、そこから隠れ状態zをサンプリングする。

入力画像のラベルone-hotベクトルから,　隠れ状態zと同じサイズのemb_labelを作る

隠れ状態zとemb_labelをconcatしたベクトルから元の入力画像を再構成するように3×256×256のTensorを作る

損失関数は入力画像と出力画像の再構成誤差(Binary-Cross-Entropy)と隠れ状態mu, varのKullback–Leibler Divergenceの和とした。

VAEGAN
--
VAEに加えて、入力画像と生成画像を識別するdiscriminatorを加えた。

discriminatorの損失関数は識別結果のBCEである。

generator(VAE)の損失関数に生成画像をdiscriminatorが入力画像と識別するかのBCEを加えた。


