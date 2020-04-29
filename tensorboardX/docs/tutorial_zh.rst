Tutorials_zh
*************

緣起
------
Google TensorFlow 附加的工具 Tensorboard 是一個很好用的視覺化工具。他可以記錄數字，影像或者是聲音資訊，對於觀察類神經網路訓練的過程非常有幫助。很可惜的是其他的訓練框架（PyTorch, Chainer, numpy）並沒有這麼好用的工具。網路上稍加搜尋可以發現已經有一些現成的套件可以讓不同的訓練框架使用 web 介面來觀察訓練情形，不過他們可以記錄的東西比較有限或是使用起來比較複雜 (tensorboard_logger, visdom)。tensorboardX 的目的就是讓其他 tensorboard 的功能都可以輕易的被非 TensorFlow 的框架使用。
目前這個套件除了 tensorboard beholder 之外支援所有 tensorboard 的紀錄型態。這個套件目前的標準測試環境為 Ubuntu 或是 Mac ，windows 則是有不定期手動測試；使用的 python 版本為 anaconda 的 python3。

安裝
-------
在命令列輸入 ``pip install tensorboardX`` 即可
或是最新版源碼安裝 ``pip install tensorboardX``

使用
-------
建立 event writer 實體
在紀錄任何東西之前，我們需要建立一個 event writer 實體。
from tensorboardX import SummaryWriter 
#SummaryWriter 是一個類別，包含這套件的所有功能。

``writer = SummaryWriter('runs/exp-1')``
#建立實體。資料存放在：``'runs/exp-1'``
#接下來要寫入任何資料都是呼叫 ``writer.add_某功能()``

``writer = SummaryWriter()``
#使用預設名稱建立實體。資料存放在：``'runs/現在時間-機器名字'`` ex. ``'runs/Aug20-obov01'``

``writer = SummaryWriter(comment='3xLR')``
#在預設資料夾後面加上註解 檔名變為：``'runs/Aug20-obov01-3xLR'``
上面的程式碼會在目前的工作目錄下建立一個叫 ``runs`` 的資料夾以及子目錄 ``exp1``。 每個子目錄都會被視為一個實驗。每次執行新的實驗時，比如說改了一些參數，這時請將資料夾重新命名，像是： ``runs/exp2``, ``runs/myexp`` 這樣可以便於比較實驗的結果。 建議：資料夾可以用時間命名或者是直接把參數當成資料夾的名稱。
建立 writer 實體之後就可以開始紀錄資料了
API 的長相大概是：``add_xxx(標籤，要記錄的東西，時間戳，其他參數)``

紀錄純量
-------------
純量是最好記錄的東西。通常我們會把每次訓練的損失記錄下來或者是測試的準確度都是值得記錄的東西。其他數據，像是學習率也值得紀錄。
紀錄純量的方法是 ``writer.add_scalar('myscalar', value, iteration)``
value 可以是 PyTorch tensor ， numpy或是 float，int 之類的python原生數字類別。

記錄影像
-------------
影像使用一個三維的矩陣來表示。這三個維度分別代表紅色，綠色，藍色的強度。一張寬200， 高100的影像其對應的矩陣大小為[3, 100, 200] （CHW）。最簡單情況是只有一張影像要存。這時候只需要注意一下是不是符合上述的規格然後將它傳到: ``writer.add_image('imresult', image, iteration)`` 即可。 
通常訓練的時候會採用批次處理，所以有一大堆影像要存。這時候請確定你的資料維度是 ``(NCHW)``, 其中 ``N`` 是batchsize。``add_image`` 會自動將他排列成適當大小。要注意的是，如果要記錄的影像是 OpenCV/numpy 格式，他們通常呈現 ``(HWC)`` 的排列，這時候要呼叫 ``numpy.transpose`` 將其轉為正確的維度，否則會報錯。另外就是注意影像的值的範圍要介於 [0, 1] 之間。 

紀錄直方圖（histogram）
-------------------------------
記錄直方圖很耗 CPU 資源，不要常用。如果你用了這個套件之後覺得速度變慢了請先檢查一下是不是這個原因。使用方法很簡單，呼叫 ``writer.add_histogram('hist', array, iteration)`` 即可紀錄。

紀錄聲音
-------------
``writer.add_audio('myaudio', audio, iteration, sample_rate)``
這功能只支援單聲道。 add_audio 要傳入的聲音資訊是個一維陣列，陣列的每一個元素代表在每一個取樣點的振幅大小。取樣頻率(sample_rate)為 44100 kHz 的情況下。一段2秒鐘的聲音應該要有88200個點；注意其中每個元素的值應該都介於正負1之間。

紀錄文字
-------------
``writer.add_text('mytext', 'this is a pen', iteration)``
除了一般字串之外，也支援簡單的 markdown 表格。

記錄網路架構。
--------------------------
(實驗性的功能，模型複雜的時候不確定對不對)
問題很多的功能。使用上比較複雜。需要準備兩個東西：網路模型 以及 你要餵給他的 tensor 
舉例來說，令模型為 m，輸入為 x，則使用方法為：
``add_graph(m, (x, ))`` 這裡使用 tuple 的原因是當網路有多個輸入時，可以把他擴充成
``add_graph(m, (x, y, z))`` ，如果只有單一輸入，寫成 ``add_graph(m, x)`` 也無妨。 
常會出錯的原因： 
- 較新的 operator pytorch本身不支援JIT
- 輸入是 cpu tensor，model 在 GPU 上。（或是反過來）
- 輸入的 tensor 大小錯誤，跑到後面幾層維度消失了
- model 寫錯，前後兩層 feature dimension 對不上
除錯方法

forward propagate 一次 ``m(x)`` 或是多個輸入時：``m((x, y, z))``
2. 用 ``torch.onnx.export`` 導出模型，觀察錯誤訊息。

高維度資料視覺化／降維 (embedding)
---------------------------------------------------
因為人類對物體的了解程度只有三維，所以當資料的維度超過三的時候我們沒辦法將他視覺化。這時候就需要降維來讓資料的維度小於等於三。降維運算由 tensorboard 以 Javascript 執行，演算法有 PCA 及 t-sne 兩種可選。這邊我們只需要負責提供每個點的高維度特徵即可。提供的格式是一個矩陣，一個 ``n x d`` 的矩陣 ``n`` 點的數量， ``d`` 是維度的多寡。 高維度特徵可以是原始資料。比如說影像，或是網路學到的壓縮結果。這原始資料決定了資料的分佈情形。如果要看得更清楚一點，你可以再傳 metadata / label_imgs 的參數進去（metadata是一個 python list 長度為 ``n``, ``label_imgs`` 是一個 4 維矩陣，大小是 ``nCHW``。這樣每個點就會有他對應的文字或圖在旁邊。不懂的話就看範例吧：https://github.com/lanpa/tensorboardX/blob/master/examples/demo_embedding.py

紀錄短片
---------------
類似於紀錄影像，不過傳入的物件維度是 ``[B, C, T ,H, W]``，其中 ``T`` 是影格的數量。所以一個 30 frame 的彩色影片 維度是 ``[B, 3, 30 ,H, W]``。

紀錄 pr curve
-------------------
根據預測的機率值以及其對應的標準答案計算 precision-recall 的結果並保存。
``add_pr_curve (tag, labels, predictions, step)``
labels是標準答案，predictions是程式對樣本的預測。 
假設有十筆資料 labels就會長得像 ``[0, 0, 1, 0, 0, 1, 0, 1, 0, 1]``，predictions則長的像 ``[0.1, 0.3, 0.8, 0.2, 0.4, 0.5, 0.1, 0.7, 0.9, 0.2]``。

pyplot 的圖表
------------------------------

用 matplotlib 畫了美美的圖表想紀錄？請用 ``add_figure`` 。傳入的物件是 matplotlib 的 figure。 
顯示結果 
Tensorboard 本質是個網頁伺服器，他讀取的資料來自於訓練網路的時候程式 (tensorboardX) 寫下的事件檔。因為 tensorboard 包含於 tensorflow，所以你需要另外安裝一份 tensorflow 在伺服器主機。我想大部分人都已經裝過了。沒裝過的話就在 unix shell 介面輸入 ``pip install tensorboard``。如果沒有使用 TensorFlow 訓練的需求，建議裝非 GPU 版本，啟動速度快得多。
接下來在命令列輸入 ``tensorboard --logdir=<your_log_dir>`` （以前面的例子來說：``tensorboard --logdir=runs``）伺服器就會啟動了。這個指令打起來很麻煩，所以我都在 ``~/.bashrc`` 加一行：``alias tb='tensorboard --logdir '`` 如此一來指令就簡化成 ``tb <your_log_dir>``。接下來就是照著終端機上的指示打開你的瀏覽器就可以看到畫面了。
