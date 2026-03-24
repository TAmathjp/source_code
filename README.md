動的モード分解（DMD）と拡張動的モード分解（EDMD）の比較

【目的】：

動的モード分解（Dynamic Mode Decomposition）と対比させて、
拡張動的モード分解 （Extended Dynamic Mode Decomposition）の意義を実装により学ぶことです。

【背景】：

Pythonの基礎を学びながら、自分の専門分野である力学系（エルゴード理論）に近くて関心のあるEDMDの理論と実装を勉強しています。
実装は、PyKoopmanというPythonライブラリの公式サイト
https://pykoopman.readthedocs.io/en/master/index.html
で学んでいます。

【工夫した点】：

コードは以下のPyKoopmanのページを参考にして書きましたが、
「1.Pythonに習熟すること」、「2.DMDとEDMD、両者の違いを学ぶこと」の２つが目的ですので、pykoopman / pydmd は使わず実装することにしました。
該当ページのリンク
https://pykoopman.readthedocs.io/en/master/tutorial_koopman_kdmd_on_slow_manifold.html

【構成の概略】：

1.　非線形な力学系（slow manifold 系）を用意する

2. そのデータから 
DMD  = 「元の座標のまま」線形モデルで近似　、
EDMD = 「特徴量に持ち上げてから」線形モデルで近似
の2つを学習する

3. 将来予測を比べて、EDMDのほうが非線形性を扱いやすいことを確認する
