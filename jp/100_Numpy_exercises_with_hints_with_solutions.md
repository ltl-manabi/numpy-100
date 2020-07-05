


# numpy 100本ノック

この「100本ノック」は、numpyのメーリングリストやStack Overflowでの議論、およびその他のドキュメントをもとに作成しました。100問に到達するために、いくつか自作した問題も含まれます。この100本ノックのゴールは、numpyの新規ユーザーにも経験者にも有用なクイックリファレンスとなること、また、numpyを教育する人が利用できる実習問題セットの提供です。

誤りを見つけたり、もっとこうした方が良いというご意見があれば、ぜひ https://github.com/rougier/numpy-100 のissueにお寄せください。
このファイルは自動的に生成されています。問題、解答、ヒントを更新するにはドキュメントを参照してください。

#### 1. numpyパッケージを、`np`という名前でインポートしてください。(★☆☆)
`ヒント: import … as`

```python
import numpy as np
```
#### 2. numpyのバージョンと設定 (Config) を表示してください。(★☆☆)
`ヒント: np.__version__, np.show_config)`

```python
print(np.__version__)
np.show_config()
```
#### 3. サイズ10のゼロ配列 (ベクトル) を作成してください。(★☆☆)
`ヒント: np.zeros`

```python
Z = np.zeros(10)
print(Z)
```
#### 4. 任意の配列を作成し、そのサイズを表示してください。(★☆☆)
`ヒント: size, itemsize`

```python
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
```
#### 5. numpyの `add` 関数についてのマニュアルをコマンドラインから参照してください。(★☆☆)
`ヒント: np.info`

```python
%run `python -c "import numpy; numpy.info(numpy.add)"`
```
#### 6. サイズ10のゼロ配列 (ベクトル) を作成し、5番目の要素に1を代入してください。(★☆☆)
`ヒント: array[4]`

```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```
#### 7. 10から49までの連番からなるベクトルを作成してください。(★☆☆)
`ヒント: arange`

```python
Z = np.arange(10,50)
print(Z)
```
#### 8. 0から49までの要素からなるベクトルを作成し、反転 (最初の要素を最後に表示) させてください。(★☆☆)
`ヒント: array[::-1]`

```python
Z = np.arange(50)
Z = Z[::-1]
print(Z)
```
#### 9. 0から8の値からなる3x3の行列を作成してください。(★☆☆)
`ヒント: reshape`

```python
Z = np.arange(9).reshape(3, 3)
print(Z)
```
#### 10. 配列 [1,2,0,0,4,0] から0でない要素のインデックスを取得してください。(★☆☆)
`ヒント: np.nonzero`

```python
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
```
#### 11. 3x3の単位行列を作成してください。(★☆☆)
`ヒント: np.eye`

```python
Z = np.eye(3)
print(Z)
```
#### 12. ランダムな値からなる3x3x3の行列を作成してください。(★☆☆)
`ヒント: np.random.random`

```python
Z = np.random.random((3,3,3))
print(Z)
```
#### 13. ランダムな値からなる10x10の行列を作成し、最小値と最大値を求めてください。(★☆☆)
`ヒント: min, max`

```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```
#### 14. ランダムな値からなるサイズ30の配列を作成し、平均値を求めてください。(★☆☆)
`ヒント: mean`

```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```
#### 15. 外周を1、内部を0とした10x10の行列を作成してください。(★☆☆)
`ヒント: array[1:-1, 1:-1]`

```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```
#### 16. 5x5の行列を作成し、周囲に0からなる要素を追加して囲んでください。(★☆☆)
`ヒント: np.pad`

```python
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)

# fancy indexingを用いる方法
Z[:, [0, -1]] = 0
Z[[0, -1], :] = 0
print(Z)
```
#### 17. 以下のコードを実行した結果はどうなるか、確認してください。(★☆☆)
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```
`ヒント: NaN = not a number, inf = infinity`

```python
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
```
#### 18. 5x5の行列を作成し、対角線から1段下がった位置の要素を1, 2, 3, 4にしてください。(★☆☆)
`ヒント: np.diag`

```python
Z = np.diag(1+np.arange(4),k=-1)
print(Z)
```
#### 19. 市松模様 (Checkerboard pattern) になるよう、8x8の行列を作成してください。(★☆☆)
`ヒント: array[::2]`

```python
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
```
#### 20. (6, 7, 8) の3次元配列について、100番目の値のインデックス (x, y, z) を求めてください。(★☆☆)
`ヒント: np.unravel_index`

```python
print(np.unravel_index(99,(6,7,8)))
```
#### 21. `tile` 関数を用いて8x8の市松模様 (Checkerboard pattern) になる行列を作成してください。(★☆☆)
`ヒント: np.tile`

```python
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)
```
#### 22. ランダムな値からなる5x5の行列の値を標準化してください。(★☆☆)
`ヒント: (x -mean)/std`

```python
Z = np.random.random((5,5))
Z = (Z - np.mean (Z)) / (np.std (Z))
print(Z)
```
#### 23. RGBAの色情報を符号なしの4バイトであらわすデータについて、独自の型を定義してください。(★☆☆)
`ヒント: np.dtype`

```python
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
```
#### 24. 5x3の行列に3x2の行列をかけてください。(実行列の積) (★☆☆)
`ヒント: np.dot`

```python
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Python 3.5以上での別解
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)
```
#### 25. 1次元配列を作成し、3より大きく、8より小さい範囲の要素について、符号を反転してください。(★☆☆)
`ヒント: >, <`

```python
# 解答例作者: Evgeni Burovski

Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)
```
#### 26. 以下のスクリプトの実行結果を確認してください。(★☆☆)
```python
# 解答例作者: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
`ヒント: np.sum`

```python
# 解答例作者: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
#### 27. 整数ベクトルZを作成し、以下のコードの実行結果を確認してください。(★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
`ヒントはありません`

```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
#### 28. 以下のスクリプトの実行結果を確認してください。
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```
`ヒントはありません`

```python
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
```
#### 29. 浮動小数点数からなる行列を作成し、整数に変換 (丸め、切り上げ、切り捨て等) してください。(★☆☆)
`ヒント: np.uniform, np.copysign, np.ceil, np.abs, np.where`

```python
# 解答例作者: Charles R Harris

Z = np.random.uniform(-10,+10,10)
print(np.copysign(np.ceil(np.abs(Z)), Z))

# 可読性は高いが効率が悪い別解
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
```
#### 30. 2つの配列に共通して出現する値を抽出してください。(★☆☆)
`ヒント: np.intersect1d`

```python
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))
```
#### 31. numpyのすべての警告を無視するにはどのようにすればよいですか? (推奨はされません) (★☆☆)
`ヒント: np.seterr, np.errstate`

```python
# Suicide modeを有効にする
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# デフォルトの状態に戻す
_ = np.seterr(**defaults)

# コンテキストマネージャを用いた実装例
with np.errstate(all="ignore"):
    np.arange(3) / 0
```
#### 32. 以下の条件式がTrueとなるか考察してください。(★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
`ヒント: 虚数`

```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
#### 33. 昨日、今日、明日の日付を表示してください。(★☆☆)
`ヒント: np.datetime64, np.timedelta64`

```python
yesterday = np.datetime64('today') - np.timedelta64(1)
today     = np.datetime64('today')
tomorrow  = np.datetime64('today') + np.timedelta64(1)
```
#### 34. 2016年7月のすべての日付を表示してください。(★★☆)
`ヒント: np.arange(dtype=datetime64['D'])`

```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
```
#### 35. オブジェクトをコピーせず上書きして、((A+B)*(-A/2)) の計算を行ってください。(A, B, Cの値は任意です)(★★☆)
`ヒント: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)`

```python
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
```
#### 36. ランダムな値 (浮動小数点数) からなる配列から、整数部だけを取り出す方法を4パターン検討してください。(★★☆)
`ヒント: %, np.floor, astype, np.trunc`

```python
Z = np.random.uniform(0,10,10)

print(Z - Z%1)
print(Z // 1)
print(np.floor(Z))
print(Z.astype(int))
print(np.trunc(Z))
```
#### 37. 各行の値が0, 1, 2, 3, 4となる、5x5の行列を作成してください。(★★☆)
`ヒント: np.arange`

```python
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)
```
#### 38. 10個の整数を生成するジェネレータ関数を定義し、それを用いた配列を作成してください。(★☆☆)
`ヒント: np.fromiter`

```python
def generate():
    for x in range(10):
        yield x

Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)
```
#### 39. 0 < Z < 1の範囲の数値からなるベクトルを作成してください。(★★☆)
`ヒント: np.linspace`

```python
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)
```
#### 40. 10個のランダムな値を生成し、昇順に並べ替えてください。(★★☆)
`ヒント: sort`

```python
Z = np.random.random(10)
Z.sort()
print(Z)
```
#### 41. `np.sum` よりも高速に小さなサイズの配列の合計を求める方法を検討してください。(★★☆)
`ヒント: np.add.reduce`

```python
# 解答例作者: Evgeni Burovski

Z = np.arange(10)
np.add.reduce(Z)
```
#### 42. 2つのランダムな値からなる配列を作成し、両者が同一であるかチェックしてください。(★★☆)
`ヒント: np.allclose, np.array_equal`

```python
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

# 許容誤差を考慮した判定
equal = np.allclose(A,B)
print(equal)

# 厳密な判定
equal = np.array_equal(A,B)
print(equal)
```
#### 43. イミュータブル (読み取り専用) な配列を作成してください。(★★☆)
`ヒント: flags.writeable`

```python
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1
```
#### 44. 直交座標系のランダムな値からなる10x2の行列について、極座標系に変換してください。(★★☆)
`ヒント: np.sqrt, np.arctan2`

```python
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
```
#### 45. ランダムな10個の値からなるベクトルを作成し、最大値を0に置き換えてください。(★★☆)
`ヒント: argmax`

```python
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```
#### 46. `x` と `y` 座標からなる構造化配列を作成し、ともに [0, 1] の範囲の値を代入してください。(★★☆)
`ヒント: np.meshgrid`

```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)
```
#### 47. 2つの配列XとYを作成し、Cauchyの行列式$C (C_{ij} = 1 / (x_{i} - y_{j}))$ を適用してください。(★★☆)
`ヒント: np.subtract.outer`

```python
# 解答例作者: Evgeni Burovski

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
```
#### 48. numpyの各スカラー型について、それぞれ使用できる最小値、最大値を表示してください。(★★☆)
`ヒント: np.iinfo, np.finfo, eps`

```python
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```
#### 49. 行列のすべての値を表示するにはどのようにすればよいですか? (★★☆)
`ヒント: np.set_printoptions`

```python
np.set_printoptions(threshold=float("inf"))
Z = np.zeros((16,16))
print(Z)
```
#### 50. ベクトルにおいて、スカラーとして与えられた任意の値に、最も近い値を表示するにはどのようにすればよいですか? (★★☆)
`ヒント: argmin`

```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```
#### 51. 座標 (position) x, yと色情報 (color) r, g, bを要素に持つ構造化配列を作成してください。(★★☆)
`ヒント: dtype`

```python
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)
```
#### 52. ランダムな値による10x2の行列を作成し、要素を座標とみなして、点間のユークリッド距離を求めてください。(★★☆)
`ヒント: np.atleast_2d, T, np.sqrt`

```python
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

# scipyによる高速な実装
import scipy
# Gavin Heverly-Coulsonの指摘に対応 (#issue 1)
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)
```
#### 53. 32bit浮動小数点数からなる配列を32bit整数に変換してください。(★★☆)
`ヒント: view and [:] =`

```python
# Thanks Vikas (https://stackoverflow.com/a/10622758/5989906)
# & unutbu (https://stackoverflow.com/a/4396247/5989906)
Z = (np.random.rand(10)*100).astype(np.float32)
Y = Z.view(np.int32)
Y[:] = Z
print(Y)

Z.astype(np.int32) # 訳注: これではダメなんでしょうか?
```
#### 54. 以下の文字列を、入力ファイルとみなして読み込んでください。(★★☆)
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```
`ヒント: np.genfromtxt`

```python
from io import StringIO

# Fake file
s = StringIO('''1, 2, 3, 4, 5

                6,  ,  , 7, 8

                 ,  , 9,10,11
''')
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```
#### 55. numpy配列において、配列の要素とインデックスを同時に取得する方法を複数検討してください。(★★☆)
`ヒント: np.ndenumerate, np.ndindex`

```python
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)

for index in np.ndindex(Z.shape):
    print(index, Z[index])
```
#### 56. 2次元 (標準) 正規分布に従う配列 (要素数10) を作成してください。(★★☆)
`ヒント: np.meshgrid, np.exp`

```python
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
```
#### 57. 2次元行列のランダムな位置の要素を1に置換してください。(★★☆)
`ヒント: np.put, np.random.choice`

```python
# 解答例作者: Divakar

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)
```
#### 58. 行列の各要素について、行の平均を引いてください。(★★☆)
`ヒント: mean(axis=,keepdims=)`

```python
# 解答例作者: Warren Weckesser

X = np.random.rand(5, 10)

# 現在のバージョンのnumpyによる実装
Y = X - X.mean(axis=1, keepdims=True)

# 古いバージョンのnumpyによる実装
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)
```
#### 59. 行列をn列目 (任意) の値でソートしてください。(★★☆)
`ヒント: argsort`

```python
# 解答例作者: Steve Tjoa

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
```
#### 60. 2次元行列の要素にヌル値 (0, None, False, NaN) があるか確認するには、どのようにすればよいですか? (★★☆)
`ヒント: any, ~`

```python
# 解答例作者: Warren Weckesser

Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())
```
#### 61. 任意の数値を定義し、配列の値の中で最も近い値を出力してください。(★★☆)
`ヒント: np.abs, argmin, flat`

```python
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
```
#### 62. イテレータを使い、(1, 3) の行列と (3, 1) の行列の和を計算してください。(★★☆)
`ヒント: np.nditer`

```python
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: 
    z[...] = x + y

print(it.operands[2])
```
#### 63. `name` 属性 (アトリビュート) を持つ配列クラスを定義してください。(★★☆)
`ヒント: class method`

```python
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
```
#### 64. 任意のベクトルについて、別のベクトルの値をインデックスとして使い、指定されるたびに、すべての要素に1を足してください (インデックスが複数回指定されうることに注意してください) 。(★★★)
`ヒント: np.bincount | np.add.at`

```python
# 解答例作者: Brett Olsen

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

# 別解
# 作者: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)
```
#### 65. ベクトルXについて、重みIを適用してカウントし、結果をベクトルFに格納してください。(★★★)
`ヒント: np.bincount`

```python
# 解答例作者: Alan G Isaac

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
```
#### 66. (dtype=ubyte) である画像データを格納した3次元行列 (w, h, 3) について、重複なく色情報を抽出してください。(★★★)
`ヒント: np.unique`

```python
# 解答例作者: Nadav Horesh

w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))
```
#### 67. 4次元行列について、3次元目と4次元目の和を1回で得るにはどのようにすればよいですか? (★★★)
`ヒント: sum(axis=(-2,-1))`

```python
A = np.random.randint(0,10,(3,4,3,4))
# タプルで次元を指定する方法 (numpy 1.7.0で導入された)
sum = A.sum(axis=(-2,-1))
print(sum)

# 2次元を1次元に展開する方法
# axisオプションを受け付けない関数を適用する場合に便利
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```
#### 68. 1次元配列Dについて、同じサイズの配列Sの値をインデックスとして用いて、抽出された要素の平均を計算してください。(★★★)
`ヒント: np.bincount`

```python
# 解答例作者: Jaime Fernández del Río

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# 参考: Pandasによるより直感的な解法
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```
#### 69. ドット積の対角成分を抽出してください。(★★★)
`ヒント: np.diag`

```python
# 解答例作者: Mathieu Blondel

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

# 実行速度が遅い解法
np.diag(np.dot(A, B))

# 実行速度が速い解法
np.sum(A * B.T, axis=1)

# 実行速度がさらに速い解法
np.einsum("ij,ji->i", A, B)
```
#### 70. ベクトル [1, 2, 3, 4, 5] について、各要素の間に0を3つ挟んで新しいベクトルを作成してください。(★★★)
`ヒント: array[::4]`

```python
# 解答例作者: Warren Weckesser

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```
#### 71. (5, 5, 3) の3次元行列について、(5, 5) の2次元行列をかけてください。(★★★)
`ヒント: array[:, :, None]`

```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```
#### 72. 行列のある行とある行を入れ替えてください。(★★★)
`ヒント: array[[]] = array[[]]`

```python
# 解答例作者: Eelco Hoogendoorn

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```
#### 73. 10組の3つの数の組み合わせを用いて、10個の (頂点を共有する) 三角形を作成し、一意な線分を抽出してください。(★★★)
`ヒント: repeat, np.roll, np.sort, view, np.unique`

```python
# 解答例作者: Nicolas P. Rougier

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```
#### 74. ある配列における値の出現回数をカウント (bincount) した結果の配列Cから、元の配列Aを復元してください。 (np.bincount(A) == C となるよう) (★★★)
`ヒント: np.repeat`

```python
# 解答例作者: Jaime Fernández del Río

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```
#### 75. 配列について、窓をスライドさせながら平均 (移動平均) を算出してください。(★★★)
`ヒント: np.cumsum`

```python
# 解答例作者: Jaime Fernández del Río

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
```
#### 76. 1次元配列Zをもとに、最初の行が Z[0],Z[1],Z[2] で、順に1つずつ要素を後ろにずらし、最後が Z[-3],Z[-2],Z[-1] になる2次元行列を作成してください。(★★★)
`ヒント: from numpy.lib import stride_tricks`

```python
# 解答例作者: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)

Z = rolling(np.arange(10), 3)
print(Z)
```
#### 77. 論理値や浮動小数点の符号を上書きで反転させてください。(★★★)
`ヒント: np.logical_not, np.negative`

```python
# 解答例作者: Nathaniel J. Smith

Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
```
#### 78. 2次元座標の組P0とP1について、対応する要素間を結ぶ直線と、任意の点pがあった時、pと直線i (P0[i],P1[i]) との距離を求めてください。(★★★)
`ヒントはありません`

```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
```
#### 79. 2次元座標の組P0とP1について、対応する要素間を結ぶ直線と、任意の座標ベクトルPがあった時、Pの各要素と直線i (P0[i],P1[i]) との距離を求めてください。(★★★)
`ヒントはありません`

```python
# 解答例作者: Italmassov Kuanysh

# 前の問題で作成したdistance関数を使用する
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
```
#### 80. 任意の行列について、指定した要素が中心になるように一部を正方行列として取り出すコードを作成してください。必要であれば、周辺のゼロ埋め (`fill`) も検討してください。(★★★)
`ヒント: minimum maximum`

```python
# 解答例作者: Nicolas Rougier

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```
#### 81. 配列 Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14] から、配列 R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]] を生成してください。(★★★)
`ヒント: stride_tricks.as_strided`

```python
# 解答例作者: Stefan van der Walt

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)
```
#### 82. 行列のランク (階数) を計算してください。(★★★)
`ヒント: np.linalg.svd`

```python
# 解答例作者: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)
```
#### 83. 配列の最頻値を求めてください。
`ヒント: np.bincount, argmax`

```python
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
```
#### 84. ランダムな値からなる10x10の行列から、すべての連続する3x3の要素をブロックとして抽出してください。(★★★)
`ヒント: stride_tricks.as_strided`

```python
# 解答例作者: Chris Barker

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)
```
#### 85. Z[i,j] == Z[j,i] となるような2次元の対称行列を作成してください。(★★★)
`ヒント: class method`

```python
# 解答例作者: Eric O. Lebigot
# 注: このコードは2次元行列でしか動作せず、値の代入にはインデックスを使用する

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```
#### 86. 複数のサイズ (n, n) の行列pと、複数のサイズ (n, 1) のベクトルpがあった場合に、行列の積の和 (テンソル積) を求めるにはどのようにすればよいですか? 結果は (n, 1) の形で出力してください。(★★★)
`ヒント: np.tensordot`

```python
# 解答例作者: Stefan van der Walt

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# 訳注: テンソル積の計算については
# https://qiita.com/nyandora/items/0fac6e307edc16c3cb91
# がわかりやすいと思います。
```
#### 87. 16x16の行列について、4x4のブロックごとの和を計算してください。(★★★)
`ヒント: np.add.reduceat`

```python
# 解答例作者: Robert Kern

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
```
#### 88. numpy配列を用いてライフゲームを実装してください。(★★★)
`ヒントはありません`

```python
# 解答例作者: Nicolas Rougier

def iterate(Z):
    # 近傍点のカウント
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
    
    # ルールの適用
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100):
    Z = iterate(Z)

print(Z)
```
#### 89. 配列から大きい順にn個の値を抽出してください。(★★★)
`ヒント: np.argsort | np.argpartition`

```python
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# 実行速度が遅い解答例
print (Z[np.argsort(Z)[-n:]])

# 実行速度が速い解答例
print (Z[np.argpartition(-Z,n)[:n]])
```
#### 90. 任意の要素数からなるベクトルについて、すべての要素、組み合わせのデカルト積を計算してください。(★★★)
`ヒント: np.indices`

```python
# 解答例作者: Stefan Van der Walt

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)
    
    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T
    
    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]
    
    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
```
#### 91. 通常の配列型からレコード型配列 (numpy.recarray) を作成してください。(★★★)
`ヒント: np.core.records.fromarrays`

```python
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
type(R)
```
#### 92. 巨大なベクトルZについて、$Z^{3}$を計算する方法を3つ検討してください。(★★★)
`ヒント: np.power, *, np.einsum`

```python
# 解答例作者: Ryan G.

x = np.random.rand(int(5e7))

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)
```
#### 93. 行列A (8,3) とB (2, 2) について、順序を無視して、Bの各行の要素を含むAの行を抽出してください。(★★★)
`ヒント: np.where`

```python
# 解答例作者: Gabe Schwartz

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```
#### 94. 10x3の行列について、行の各要素の値が等しくない行だけを抽出してください。(例: [2, 2, 3]) (★★★)
`ヒントはありません`

```python
# 解答例作者: Robert Kern

Z = np.random.randint(0,5,(10,3))
# 訳注: 何も返ってこない (=各要素の値が等しい) データ
# Z = np.ones((10, 3))
print(Z)
# あらゆるdtypesに対応した解答例 (文字列型やレコード型含む)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
# 数値型のみに対応した解答例
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)
```
#### 95. 整数ベクトルを2進数表記の行列 (各ビットが行列の要素に対応) に変換してください。(★★★)
`ヒント: np.unpackbits`

```python
# 解答例作者: Warren Weckesser

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# 解答例作者: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```
#### 96. 2次元行列から行の重複 (同じ要素からなる行) を除去してください。(★★★)
`ヒント: np.ascontiguousarray | np.unique`

```python
# 解答例作者: Jaime Fernández del Río

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

# 解答例作者: Andreas Kouzelis
# NumPy >= 1.13
uZ = np.unique(Z, axis=0)
print(uZ)
```
#### 97. 2つのベクトルA, Bについて、内積、外積、和、乗算の処理を縮約記法を使って記述してください。(★★★)
`ヒント: np.einsum`

```python
# 解答例作者: Alex Riley
# Make sure to read: http://ajcr.net/Basic-guide-to-einsum/

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```
#### 98. 2つのベクトルX, Yで表現される曲線について、等間隔にサンプリングしてください。(★★★)
`ヒント: np.cumsum, np.interp`

```python
# 解答例作者: Bas Swinckels

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)
```
#### 99. 整数nと2次元行列Xについて、次数nの多項分布からサンプリングされたと見なせるような行を抽出してください。次数nとは、抽出した値が整数のみからなり、合計するとnになるという意味です。(★★★)
`ヒント: np.logical_and.reduce, np.mod`

```python
# 解答例作者: Evgeni Burovski

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```
#### 100. 1次元配列Xについて、ブートストラップ法でN回サンプリングし、平均の95%信頼区間を求めてください。(★★★)
`ヒント: np.percentile`

```python
# 解答例作者: Jessica B. Hamrick

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```