原文：http://www.jianshu.com/p/ab1a003f2275

深度学习里面有很多看起来很简单但是实际却有大用场的算法。Autoencoder作为其中的一种就是。作为一种无监督学习的手段，autoencoder在维度灾难里为数据降维有着深远的意义。

什么是Autoencoder呢？我大概的理解是这样的，比如说我们提取一张500x500的彩色图片的时候，按照每个像素来算就有500x500x3（RGB颜色）=750000个像素，这么多的像素就得对应这么多的权重，如果再是大一点的数据那训练的时候用的资源就海了去了。但是，图片里每一个像素都是有用的么？不尽然，比如我想要一个模型，来感知一张人脸是开心还是不开心，除了面部的那些像素，其余的很多像素都是浪费的，或者说对判别人的表情不是那么重要的，而且这些有用的像素分布在一个三维的空间里，就像一大袋米里面的几颗绿豆，这种稀疏性也会给训练带来不必要的麻烦。那我怎么能把图中的信息浓缩起来？用极少甚至一个维度的向量就能表示最重要的信息呢？

Autoencoder就是很好的一个工具。给一个图。

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/616445-5073f1341d109e92.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
假设我们用cnn的方法对一张图做autoencoding, 大家知道cnn网络是提取图里的有效信息，pooling作为一种降维的手段浓缩信息。大概浓缩了几次之后，形成中间那短短的一条，我们就叫它羞羞的棒棒吧（因为它没别人长，哈哈哈哈）。 它不是真短，可以很长，但是比起原本输入的数量那是要短的多。好，浓缩的这一边我们叫它encoding。在另一边，有个做完全相反的事情，就是从这条羞羞的棒棒逐步撑回到原来图片大小的维度，得到一个输出。这个输出会和输入的图片做对比，如果有差异，做反向传播得到新权重。最终得到的这个羞羞的棒棒里的权重值，就是原图的浓缩精华。

“我对着镜子做一个惟妙惟肖的小泥像。” 大概这个意思。 这条羞羞的棒棒是很厉害的，不仅能浓缩精华，还能记录一些少量的差异。比如输入的图像有些噪点，对比的图像是高清，这个羞棒可以记录下噪点到高清的逻辑关系，理论上就能解马赛克啦（哈哈哈）。同样，如果输入是黑白照片，训啦的对比图是彩色照片，那这个羞棒可以描述照片上黑白到彩色的关系，就能给他没见过的黑白照片上色啦。

扯太多了，直接上code，还是用keras。
我的训练数据是前阵子做kaggle上辨别猫狗的训练数据,[这里](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)可以下到，人太懒，懒得找其他数据，没办法。
```python
ROW = 80
COL = 80
CHANNELS = 3

TRAIN_DIR = 'cat_dog_train/'
TEST_DIR = 'cat_dog_test/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] #读取数据

def readImg(imgFile):
    colorImg = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    colorImg = cv2.resize(colorImg, (ROW, COL))/255.0
    greyImg = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    greyImg = cv2.cvtColor(greyImg,cv2.COLOR_GRAY2RGB)
    greyImg = cv2.resize(greyImg, (ROW, COL))/255.0
    return greyImg,colorImg

def generateDate(imgFiles):
    count = len(imgFiles)
    dataX = np.ndarray((count, ROW, COL,CHANNELS), dtype=float)
    dataY = np.ndarray((count, ROW, COL,CHANNELS), dtype=float)
    for i, image_file in enumerate(imgFiles):
      gImg,cImg = readImg(image_file)
      dataX[i] = gImg
      dataY[i] = cImg  
    return dataX,dataY

import math
def chunked(iterable, n):
    chunksize = int(math.ceil(len(iterable) / n))
    return (iterable[i * chunksize:i * chunksize + chunksize]
            for i in range(n))

BATCHNUMBER= 20
chuckedTrainList = list(chunked(train_images,BATCHNUMBER))

# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
random.shuffle(train_images)
```
两个准备数据的方法，第一个是把每张训练用的图片，转一份拷贝为黑白的，另一张彩色的用作对比Y。注意下，openCV这个鬼东西读入图片默认是BRG。。需要转一下成RGB，不然显示图片的时候会出错，对了 别忘了除上255做归一化。第二个方法不用解释了，第三个方法是用来吧数据分割成几块做训练。

下面到了最激动人心的搭建模型时刻
```python
baseLevel = ROW//2//2
input_img = Input(shape=(ROW,COL,CHANNELS))
x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Flatten()(x)

encoded = Dense(2000)(x)
one_d = Dense(baseLevel*baseLevel*128)(encoded)
fold = Reshape((baseLevel,baseLevel,128))(one_d)

x = UpSampling2D((2, 2))(fold)
x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```
input 是一张图片80x80的像素，做两层卷积的操作做加上两次池化，每次长宽减半，到最后图就剩下20x20了。再把这20x20xfilter#(128)个元素拉直，传入传说中羞羞的棒子（也不短啦其实，有2000呢）。这个羞羞的棒子就成为了一张图片的精华！ 下面的代码就是上面部分的反操作。最后用sigmoid函数激活。简单明了。

```python
checkpoint = ModelCheckpoint(filepath="path/weights.hdf5", verbose=1, save_best_only=True)
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

cbks = [checkpoint,earlyStopping]

digitalImg = dict()
epoch = 500

for ii in range(epoch):
    i = ii%len(chuckedTrainList)
    print("epoch =====>"+str(ii))
    imgfiles = chuckedTrainList[i]
    if str(i) in digitalImg:
        dataX = digitalImg[str(i)][0]
        dataY = digitalImg[str(i)][1]
    else:
        dataX,dataY = generateDate(imgfiles)
        digitalImg[str(i)] = (dataX,dataY)
        
    autoencoder.fit(dataX, dataY,
                nb_epoch=1,
                batch_size=50,
                shuffle=True,
                verbose=2,
                validation_split=0.3,
                callbacks=cbks)
```
接下来就开始训练了，定500个epoch。两个callback蛮有用的，一个是当训练的过程有大的进步，就把最好的训练成果备份下来。另一个是当训练不再提升了，自动结束训练。


最后，我们来看看结果：
#####这是10个epoch产生的效果，卷积核3：
![Paste_Image.png](http://upload-images.jianshu.io/upload_images/616445-c1f1e129c73ad4ff.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最左边的图是模型从来没见过的黑白照片，中间这张是模型根据训练的羞羞棒棒给出的预测结果，右边的是做对比的彩色原图。

10个epoch的训练太少了，但是大概能看出那么点意思来，很多图片里的细节被忽略掉了，颜色也和沙盘一样洒满一地。

#####400个epoch，卷积核3：

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/616445-7ff9de05f309a505.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

现在照片已经有轮廓了，至少狗的形状都能看见了。颜色嘛没有那么明显，但是模型大概知道应该往哪儿填和填什么了。

#####400个epoch 卷积核改成5：
改成5后每个卷积核收集到的信息更多，能抓取更多的像素间的关系。效果是这样的。

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/616445-8f339abed5e9f34d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
看这个狗已经可以很看的清楚轮廓了，模型知道把周围环境的颜色和狗给区分开来。

#####400个epoch 卷积核5 卷积层改为256 输出：
原来的卷积层只有128的输出，这一步加了一倍，就是能提供给模型分析更多features的空间多了一倍。但已经是我可怜的GPU的极限了，话说赞一下我的GPU，文能打守望，武能算模型，全能小天才啊有没有。
![Paste_Image.png](http://upload-images.jianshu.io/upload_images/616445-3df393a308e169e3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
到目前已经有更多的细节了，颜色也越来越能分得清楚了。最后一张狗简直在卖萌啊有木有。

总结：花了很多时间在调参数上，最后一批的结果算是在我硬件能力范围比较好的结果了。Autoencoding真的是一项很牛逼的技术，简单直接。但是也有一定的局限性。 它只考虑了重建误差最小，而一张图片的颜色和图片的实质内容是有关联的，这种信息不能被抓捕到, 最近很火的对抗网络能从某种程度上很大的解决这个问题，所以用对抗网络对照片填色这个工作有很好的效果。 另一个原因，也可能是我的训练数据集实在是太小了，只有25000张训练图，新照片上很多信息模型都无法知道是什么，所以无法填色。 最后，再赞一下我的GPU，你要是内存再大些就好了。。。
