再構築誤差による画像分類モデル
=
入力した画像をオートエンコーダモデルで再構築し、その出力と入力画像の差(再構築誤差)を基に画像の分類を行う。

データセット
=
Yahoo!の画像検索エンジンで検索した結果の画像をデータセットの基とした。

その画像に対して、opencvの顔検出を用いて顔にフォーカス、(256, 256)にリサイズした画像を生成する。これをデータセットとした。

モデル構成
=
入力と出力は256×256の画像のRGB(3チャネル)

(モデルの画像)

