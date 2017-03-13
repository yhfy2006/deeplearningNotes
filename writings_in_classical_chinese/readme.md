原文：http://www.jianshu.com/p/dbab07cefbd2


刚开始学习RNN的时候利用英文文本训练写英文的文章。出于练手的目的写一个文言文的模型的练习。

本模型利用了一个简单的LSTM模块，即带记忆的RNN。虽然LSTM也有类似RNN的局限性，但是训练一个小型的中文文本应该是没有问题。

快速复习下LSTM原理：

![LSTM3-var-GRU.png](http://upload-images.jianshu.io/upload_images/616445-cb7fe1de7a602963.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



为了快速搭建，我选用了Keras，不多说，快速搭建模型神器。

第一步数据处理，就是把每一个出现过的字都标上一个index
```
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np
import random
import sys
import collections

#我们这里用的训练文本是《项羽本纪》
path = "xiangyubenji.txt"
text = open(path).read()
print('corpus length:', len(text))
#corpus length: 11247
print(text[:10])
#项籍者，下相人也，字

chars = sorted(list(set(text)))
print('total chars:', len(chars))
#total chars: 1075
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
```

我们来定义下每次训练文本的句长。这里定义40个字为一句吧。
step为3，下一句是3个字以后。其实可以自由定义。
简单的来说训练的X就是一句话
训练的Y就是这句话结束后的下一个字。

```
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
    
print('nb sequences:', len(sentences))
print(sentences[:1])
print(next_chars[:1])
#nb sequences: 3736
#['项籍者，下相人也，字羽。初起时，年二十四。其季父项梁，梁父即楚将项燕，为秦将王翦']
#['所']
```

我们把每句话每个字都向量话。我们用1个1075个元素的向量来表示一个字，按每个字的index在那个位置标记1(不懂参考one hot encoding)。所以一句话就要 40 x 1075 个元素。
```
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
```
建立模型,非常简单，只用一层LSTM模块，输出为128. 单次步长为一句话的长度，输入为1075.  input_shape=(maxlen, len(chars))

```
# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

接下来我们来训练这个模型，来60次epoch。
每次10训练完之后 打出来看看效果。随机从文中选择一句话开始，打出后面预测的四百个字。

```
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    if iteration % 10 == 0:
        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [1.2]: 
        # 在sample的时候选择的分布可选值[0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- 用这句话来开头写个故事: "' + sentence + '"')
            print('----- 你准备好了，我开始瞎扯了:')
            print()
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                #大概的原理是从概率到index的映射范围
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

其实就这么点。我们来看看效果。

#####[epoch10]：
**西，必举秦矣。故不如先斗秦赵。夫被坚执锐，义不如公；坐而运策，公不如义。”因下令沛公曰‘曰“沛公盖盖亦我逐其宝”反，魏沛公与以责令行，常何且至时，项异楚也。”於是项项王十壁遂沛诛遂诛。财项伯亦郡项伯徙约地击，七史披史者事，大咸王马乌武沛与楚乃信项孰闻初亦，行阻陈县而侍者？不他行杀塞馀何；为期曹保子胜高齐当时为项梁射故项人相来，三素魏人亡城不北。田荣言与阿沛公。项大破与江召益夜反有详汉”军淮骂马稍哀将屠，而免立司而楚将军，通办得；罪汉入城胜诸，我中别可战罪若兵於言当高。沛苛如、谨往骑去万耳景卮万人武夫山将有举，具有安广剑国为项王内并赵彭，战败灭：其粮，多二用也。”沛胜即天下相围之公相项羽谓自曰当之不敢数略诛。不使者名可知遂不为其耳。项羽大久汉曰：“首项氏与焉项积曰：“粮数不然。籍於是项王谓羊欲立国诛岁卒，项王引姻受右，欲为遇披从而北者於义者。今不败从所属於行，遂霸於田荣不免。恐恐复可胜又关处，危贵籍闻。韩及然人，乃应渡河击东意，公就‘得哙以其臣，诸侯侯诸将六出数高。项梁**

#####[epoch20]：
**无西意，而北击齐。徵兵九江王布。布称疾不往，使将将数千人行。项王由此怨布也。汉之。樊哙周项王。
　　项约军走引兵渡三金，亦言。城数百项王，都鸿万数伤降入，复烦王其秦，遂去四北陵之。项王瞋之脱逐。汉王弟南数千壁遂烦，瞳三胡万人，使人追说汉王，非从一为万行。。项弟氏，阕之数上。
　　居数百百言，数百壁项羽追军，此亦分至，汉王皆为怜侯。长婴说为骑齐王。齐人而射数宝项王。项羽急，曰：“诺。相大杓猴乎？”曰：“心为冻信乘，闻楚皆则兵属秦常，军咸阳至久都也。
　　西时，不当成金从武侯。项王如史杀之，年。诸侯至范陵，走项王。项氏、鲁中，闻皆走下，引兵西。汉王徇鲁，项王。项羽天下彭项王。军战而北与西，封又。关项王，乃散知代王。长怀军，项王。项羽渡沛事骑，使人追忍沛。项羽谓无事之，知张矣。汉王因玉力，水走。、臣江桓张马章？”项氏则功之，走楚楚而立秦嘉霸矣。今者攻罪关，诸将军薛秦万为骑城。籍俱太曹无益子意，皆复南声行万人，得吴南南月必人下属与籍倍侯。诸侯吏相曰：“虽吏吏卒，使项王乃自合马**

#####[epoch30]：
**帝，曰：“古之帝者地方千里，必居上游。”乃使使徙义帝长沙郴县。趣义帝行，其群臣稍”背齐数，
　　时时，兵汉梁患年，项羽乃召彭越王。汉王击秦嘉，以恶侯与齐与齐。齐，南功於项王曰：“曰：“起高”何能胜亡走我何弃：“城军荥，诛非游食，封项羽为诗将章卒者、东安子北有前，寿金奴、汉兵四力者秦军，项王乃相置汉数十、睢其以至韩用，得楚十四，故天亡至，观破秦入，斩西。长公已、，间与韩游战，近令其心为地，荥又不人，封人军中，扞内未江，北数由阳关，故立为上为长君遗为封武，尝从古汉，可与彭战，可为独将，须又不重。欣月走胜故益为翟为王王患来，初之。知者高！夫角走楚，楚楚之出，睢有所知，王曰：“，为王将，下。今岁走城汉甬，此秦战，破秦易。於。王闻所诺。汉於歇汉王，得杀数十，从心长史若。项羽乃定陶怀王，汉王乃使徙瞳汉王父亦叱、分烧，妇为常，收其怒，杀之卒上。”乃谓、与梁
　　当汉王有砀成皋。长，欲汉，如公十复数侯。约欲攻立事立诛为诸君王，封令卫为前，郎王数为沛公沛军事而塞王。”乃即项王未：“吾闻**

#####[epoch60]：
**张良出，要项伯。项伯即入见沛公。沛公奉卮酒为寿，约为婚姻，曰：“吾入关，秋豪不敢；。如用之帝者，，大汉王？”乃曰：“巴、间以楚，追九军至，骑何阳，长史若者，不敢出。公。
　　沛公先为封项王曰：“彭越左劫积，楚王急，乃城之胶无不至江阳，定陶所知。陈馀受东与韩而必公。”汉王听，诫汉王至，披王定陶报项王。项王使沛公军项王闻之王三年战阳。项王乃自王为项王乃疑沛公。项羽乃召沛公曰：“吾入急，而恐，至破中，使者高行，故楚於张耳。项王乃相引兵四面王。项王渡侵，急下，尽降不别，左勿楚北，闻羽引欲立汉王父後以来出女。项王乃驰行，斩睢，故八君。”项王即之，曰：“旦有不趣。”项王曰：“田角项羽大破曰曰：“谨诺楚以楚，巴得楚之楚破之。章曰：“章邯乃魏项王项王伏，置军，无以至东定，项梁怒曰：“田阳为齐受天下，使人新六巩此，王大怒。曰：“将戮力也也”，又以即其意，不应大，与大夫俱齐。。项梁闻沛公左，项王乃疑陈留沛公与籍斩未，使者起，项王伏栎芮欣，都废；不义头。
　　项王闻汉羽欲得与项羽大惊曰：“**


总结：可以大概看出来，越到后来学到的专用名词越多。所以效果还是蛮好的。反正比我写的好多了。
