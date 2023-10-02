需要安装

torch\transformers\sofa

nltk\boto3



# metrics

基于L2距离的暴力全量搜索

```python
import faiss
# build the index， d is dimension
index = faiss.IndexFlatL2(d)
print(index.is_trained)        # IndexFlatL2不需要训练过程
# add vectors to the index
index.add(database)                 
print(index.ntotal)

# get nearest k neighbors
k = 4                          
D, I = index.search(queries, k)     # search
print(I[:5])                   # neighbors of the 5 first queries
print(D[:5])                   # distance of the return results
```



# dataset

XTREME

从git上下载了XTREME项目，然后使用git bash运行下载数据对应的命令

git bash中需要先正确激活conda环境

```bash
# 先尝试conda list看conda是否正常使用
conda list

# 一般来说需要source，否则不能使用activate指令
source activate
conda activate your_env

# 运行EXTREME项目中给出的指令
bash install_tools.sh
bash scripts/download_da

```


在bucc上的测试结果不太理想，zh-en
```shell
# 使用第八层
load ./dataset/bucc/bucc2017/zh-en/XLM-R_zh_8.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R_en_8.npy
zh-en training gold has: 1899
candidates generated and similarity generated
[[68231, 45952, -0.2085723876953125], [46995, 26687, -0.080535888671875], [25526, 40949, -0.07354736328125], [82003, 40949, -0.0593719482421875], [62974, 38469, -0.03363037109375]]
optimized bucc f1: 0.010151653269246703, threshold: -0.00018310546875

# 使用12层
load ./dataset/bucc/bucc2017/zh-en/XLM-R_zh_12.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R_en_12.npy
zh-en training gold has: 1899
candidates generated and similarity generated
[[32660, 34964, -31.797760009765625], [54497, 84696, -9.130340576171875], [48350, 21782, -8.371826171875], [9648, 14358, -7.8807373046875], [87419, 49028, -7.798980712890625]]
optimized bucc f1: 0.007790162847553142, threshold: -0.195587158203125

(ljh_faiss) zhangrichong@gpu-15:~/data/ljh/multilingual$ python evaluate_tools.py 

 evaluate on XLM-R, layer-8 in fr-en setting
load ./dataset/bucc/bucc2017/fr-en/XLM-R_fr_8.npy, ./dataset/bucc/bucc2017/fr-en/XLM-R_en_8.npy
fr-en training gold has: 9086
candidates generated and similarity generated
[[9849, 257387, -0.183807373046875], [254549, 138997, -0.1254730224609375], [197207, 316489, -0.086639404296875], [213217, 316489, -0.0865020751953125], [207422, 176866, -0.0765838623046875]]
optimized bucc f1: 0.03557801822323462, threshold: -0.0

 evaluate on XLM-R, layer-12 in fr-en setting
load ./dataset/bucc/bucc2017/fr-en/XLM-R_fr_12.npy, ./dataset/bucc/bucc2017/fr-en/XLM-R_en_12.npy
fr-en training gold has: 9086
candidates generated and similarity generated
[[173799, 22683, -16.76226806640625], [209519, 117624, -13.99072265625], [242470, 197796, -10.548919677734375], [72941, 95707, -8.189727783203125], [193302, 25840, -8.031463623046875]]
optimized bucc f1: 0.009546539379474939, threshold: -0.642669677734375


```

```shell
(ljh_faiss) zhangrichong@gpu-15:~/data/ljh/multilingual$ python evaluate_tools.py 

 evaluate on XLM-R, layer-7 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R_de_7.npy, ./dataset/bucc/bucc2017/de-en/XLM-R_en_7.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[320371, 120504, -0.003371715545654297], [213787, 267019, -0.0032634735107421875], [186434, 174066, -0.002659320831298828], [308087, 187099, -0.0024547576904296875], [168456, 381266, -0.0023746490478515625]]
optimized bucc f1: 0.004515301724646888, threshold: -0.0

 evaluate on XLM-R, layer-8 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R_de_8.npy, ./dataset/bucc/bucc2017/de-en/XLM-R_en_8.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[212835, 304824, -0.0725555419921875], [103827, 203361, -0.068359375], [375159, 272779, -0.06060791015625], [22418, 398893, -0.0562286376953125], [335162, 112788, -0.0552215576171875]]
optimized bucc f1: 0.018273747251735153, threshold: -0.0

 evaluate on XLM-R, layer-12 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R_de_12.npy, ./dataset/bucc/bucc2017/de-en/XLM-R_en_12.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[376621, 202917, -22.16497802734375], [251671, 18112, -9.55303955078125], [226756, 261314, -9.12115478515625], [51009, 381015, -8.712677001953125], [259195, 18112, -8.624786376953125]]
optimized bucc f1: 0.011703888779994757, threshold: -0.0

 evaluate on XLM-R, layer-7 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R_ru_7.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R_en_7.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[216213, 172903, -0.0015544891357421875], [7054, 549013, -0.0013284683227539062], [441894, 494588, -0.0005350112915039062], [265479, 381909, -0.0003981590270996094], [204072, 381909, -0.0003795623779296875]]
optimized bucc f1: 0.005260245379926483, threshold: -6.198883056640625e-06

 evaluate on XLM-R, layer-8 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R_ru_8.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R_en_8.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[354455, 366776, -0.2178192138671875], [180573, 123134, -0.1841583251953125], [394285, 278068, -0.13446044921875], [122394, 400751, -0.1253204345703125], [121546, 484161, -0.1134033203125]]
optimized bucc f1: 0.02191517967038863, threshold: -3.0517578125e-05

 evaluate on XLM-R, layer-12 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R_ru_12.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R_en_12.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[338310, 444314, -7.202850341796875], [85982, 401896, -6.362701416015625], [390839, 534536, -6.153045654296875], [426578, 437634, -6.1051025390625], [180573, 206701, -5.680267333984375]]
optimized bucc f1: 0.01822133663814235, threshold: -0.127655029296875
```


## XLMR-LARGE
结果也很不好
```shell
evaluate on XLM-R-LARGE, layer-1 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_zh_1.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_en_1.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
[[70538, 35698, -1.53631591796875], [32556, 67541, -1.48944091796875], [69879, 61060, -1.48614501953125], [13832, 630, -1.46533203125], [32554, 35698, -1.44696044921875]]
optimized bucc f1: 0.00014962220393506393, threshold: -0.4783935546875

 evaluate on XLM-R-LARGE, layer-5 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_zh_5.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_en_5.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
[[89773, 35564, -139.18267822265625], [89772, 35564, -128.54534912109375], [89768, 35564, -120.00494384765625], [61311, 35564, -101.76568603515625], [84870, 35564, -99.318115234375]]
optimized bucc f1: 0.0, threshold: 0.0

 evaluate on XLM-R-LARGE, layer-10 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_zh_10.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_en_10.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
[[89773, 35564, -67.19989013671875], [89772, 35564, -65.96270751953125], [68276, 78585, -63.3292236328125], [32558, 43032, -59.45855712890625], [89768, 35564, -57.31524658203125]]
optimized bucc f1: 0.0, threshold: 0.0

 evaluate on XLM-R-LARGE, layer-15 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_zh_15.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_en_15.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
[[53982, 48403, -122.2401123046875], [86638, 48403, -120.482421875], [41680, 48403, -117.2513427734375], [67936, 48403, -117.2374267578125], [31479, 48403, -115.17779541015625]]
optimized bucc f1: 0.0, threshold: 0.0

 evaluate on XLM-R-LARGE, layer-20 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_zh_20.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_en_20.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
[[74501, 35345, -36.1458740234375], [89051, 43025, -34.64794921875], [68276, 35564, -33.3399658203125], [22270, 61951, -31.4876708984375], [24889, 13463, -30.9931640625]]
optimized bucc f1: 7.308605883427736e-05, threshold: -14.81689453125

 evaluate on XLM-R-LARGE, layer-24 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_zh_24.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R-LARGE_en_24.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
[[90655, 74547, -21.0865478515625], [35243, 74547, -20.84112548828125], [77831, 52010, -18.1689453125], [50496, 2507, -18.01995849609375], [81362, 23762, -15.51806640625]]
optimized bucc f1: 0.0006115563053563897, threshold: -0.8734130859375

 evaluate on XLM-R-LARGE, layer-1 in fr-en setting
load ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_fr_1.npy, ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_en_1.npy
fr-en training gold has: 9086
doc sent_num:369810, query sent_num:271874
candidates generated and similarity generated
[[253426, 166025, -1.1531982421875], [141736, 151481, -1.0281982421875], [169145, 103342, -0.96807861328125], [38819, 139394, -0.93682861328125], [89361, 92220, -0.9290771484375]]
optimized bucc f1: 0.00043996656254124684, threshold: -0.41546630859375

 evaluate on XLM-R-LARGE, layer-5 in fr-en setting
load ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_fr_5.npy, ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_en_5.npy
fr-en training gold has: 9086
doc sent_num:369810, query sent_num:271874
candidates generated and similarity generated
[[264237, 202244, -75.7666015625], [200432, 130254, -68.0009765625], [220243, 171631, -67.103515625], [215383, 127462, -59.46044921875], [214646, 33228, -56.182373046875]]
optimized bucc f1: 8.71763577717723e-05, threshold: -18.6878662109375

 evaluate on XLM-R-LARGE, layer-10 in fr-en setting
load ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_fr_10.npy, ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_en_10.npy
fr-en training gold has: 9086
doc sent_num:369810, query sent_num:271874
candidates generated and similarity generated
[[200432, 33974, -43.37506103515625], [73857, 151432, -42.5162353515625], [77342, 115798, -41.7305908203125], [65654, 197126, -40.64654541015625], [264237, 71411, -40.505615234375]]
optimized bucc f1: 2.1355353075170844e-05, threshold: -0.0

 evaluate on XLM-R-LARGE, layer-15 in fr-en setting
load ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_fr_15.npy, ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_en_15.npy
fr-en training gold has: 9086
doc sent_num:369810, query sent_num:271874
candidates generated and similarity generated
[[200432, 367324, -69.2679443359375], [23627, 323678, -64.50872802734375], [128077, 273421, -61.1920166015625], [77342, 115798, -58.0933837890625], [53828, 323678, -57.522216796875]]
optimized bucc f1: 2.1355353075170844e-05, threshold: -0.0001220703125

 evaluate on XLM-R-LARGE, layer-20 in fr-en setting
load ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_fr_20.npy, ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_en_20.npy
fr-en training gold has: 9086
doc sent_num:369810, query sent_num:271874
candidates generated and similarity generated
[[149621, 102336, -117.504638671875], [103116, 260272, -54.8348388671875], [23627, 153457, -33.68994140625], [251222, 207420, -28.8902587890625], [200432, 327719, -23.5301513671875]]
optimized bucc f1: 2.1355353075170844e-05, threshold: -0.0

 evaluate on XLM-R-LARGE, layer-24 in fr-en setting
load ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_fr_24.npy, ./dataset/bucc/bucc2017/fr-en/XLM-R-LARGE_en_24.npy
fr-en training gold has: 9086
doc sent_num:369810, query sent_num:271874
candidates generated and similarity generated
[[129557, 113207, -74.7080078125], [16720, 113207, -56.76617431640625], [53846, 345352, -56.6790771484375], [121253, 30820, -42.8282470703125], [15441, 40190, -36.066650390625]]
optimized bucc f1: 0.00019013214183857778, threshold: -6.85205078125

 evaluate on XLM-R-LARGE, layer-1 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_de_1.npy, ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_en_1.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[394731, 168573, -1.1678466796875], [93721, 94017, -1.05963134765625], [205330, 73579, -0.96746826171875], [175887, 178034, -0.96466064453125], [63341, 303033, -0.948486328125]]
optimized bucc f1: 0.0002886107002417115, threshold: -0.446044921875

 evaluate on XLM-R-LARGE, layer-5 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_de_5.npy, ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_en_5.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[127078, 206291, -79.63421630859375], [182593, 206291, -76.5611572265625], [169719, 206291, -70.04754638671875], [386993, 246316, -66.98138427734375], [248453, 206291, -66.8138427734375]]
optimized bucc f1: 1.4169356876506971e-05, threshold: -0.0

 evaluate on XLM-R-LARGE, layer-10 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_de_10.npy, ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_en_10.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[189148, 190300, -63.56817626953125], [345091, 172854, -52.3143310546875], [248453, 92337, -50.12078857421875], [127078, 126362, -49.7952880859375], [237191, 381007, -46.4007568359375]]
optimized bucc f1: 1.4169356876506971e-05, threshold: -0.0

 evaluate on XLM-R-LARGE, layer-15 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_de_15.npy, ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_en_15.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[189148, 145382, -74.043212890625], [301057, 181908, -71.7371826171875], [187508, 135036, -61.64739990234375], [182593, 351150, -60.76922607421875], [216625, 135036, -60.15545654296875]]
optimized bucc f1: 1.4169356876506971e-05, threshold: -0.0

 evaluate on XLM-R-LARGE, layer-20 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_de_20.npy, ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_en_20.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[169649, 65748, -40.0228271484375], [301057, 181908, -21.73779296875], [189148, 145382, -21.195556640625], [187508, 273000, -19.85888671875], [17729, 102922, -18.4150390625]]
optimized bucc f1: 1.4169356876506971e-05, threshold: -0.0

 evaluate on XLM-R-LARGE, layer-24 in de-en setting
load ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_de_24.npy, ./dataset/bucc/bucc2017/de-en/XLM-R-LARGE_en_24.npy
de-en training gold has: 9580
doc sent_num:399337, query sent_num:413869
candidates generated and similarity generated
[[76876, 316220, -60.4730224609375], [156327, 316220, -48.41619873046875], [70096, 212642, -35.63720703125], [191358, 136370, -33.5025634765625], [244080, 136370, -31.875732421875]]
optimized bucc f1: 0.0006759611322348965, threshold: -4.99810791015625

 evaluate on XLM-R-LARGE, layer-1 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_ru_1.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_en_1.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[228840, 243463, -1.04779052734375], [323303, 394224, -1.02386474609375], [357735, 115586, -1.023681640625], [128534, 316256, -1.02362060546875], [182233, 326842, -1.01617431640625]]
optimized bucc f1: 9.095870474804439e-05, threshold: -0.6116943359375

 evaluate on XLM-R-LARGE, layer-5 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_ru_5.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_en_5.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[413600, 521925, -134.5421142578125], [295388, 446369, -115.39178466796875], [320563, 26919, -114.81695556640625], [295385, 94363, -109.1949462890625], [53793, 394149, -108.739013671875]]
optimized bucc f1: 1.5019187011407074e-05, threshold: -17.72235107421875

 evaluate on XLM-R-LARGE, layer-10 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_ru_10.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_en_10.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[413600, 89289, -134.03271484375], [413588, 86200, -105.036865234375], [413584, 430630, -96.9432373046875], [413587, 86200, -91.94219970703125], [53793, 451091, -91.51263427734375]]
optimized bucc f1: 0.0, threshold: 0.0

 evaluate on XLM-R-LARGE, layer-15 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_ru_15.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_en_15.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[413584, 86200, -134.651123046875], [413600, 86200, -133.20611572265625], [295388, 419198, -118.136474609375], [295385, 419198, -112.77606201171875], [413588, 86200, -111.1092529296875]]
optimized bucc f1: 0.0, threshold: 0.0

 evaluate on XLM-R-LARGE, layer-20 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_ru_20.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_en_20.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[413584, 392800, -50.27783203125], [413600, 86200, -37.926513671875], [295388, 419198, -37.5916748046875], [53793, 343517, -37.30224609375], [413497, 419198, -36.7218017578125]]
optimized bucc f1: 0.0, threshold: 0.0

 evaluate on XLM-R-LARGE, layer-24 in ru-en setting
load ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_ru_24.npy, ./dataset/bucc/bucc2017/ru-en/XLM-R-LARGE_en_24.npy
ru-en training gold has: 14435
doc sent_num:558401, query sent_num:460853
candidates generated and similarity generated
[[135900, 260842, -390.55126953125], [406761, 547257, -86.0743408203125], [217681, 376412, -81.0115966796875], [249187, 276264, -63.0892333984375], [293997, 335759, -61.99786376953125]]
optimized bucc f1: 0.001400093620718012, threshold: -0.46929931640625

```


使用mean pool之后效果好了很多
```shell
(ljh) zhangrichong@gpu-15:~/data/ljh/multilingual$ python bucc_evaluate.py --flag
reading bucc zh-en, total lines: 94637 88860
read bucc file zh-en, total lines:(94637, 88860)
tokenizer.pad_token=<pad>, pad_token_id=1
Some weights of the model checkpoint at ./model/xlm-r/ were not used when initializing XLMRobertaModel: ['lm_head.layer_norm.bias', 'lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.decoder.weight', 'lm_head.dense.bias', 'lm_head.dense.weight']
- This IS expected if you are initializing XLMRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
run inference sentence:94637, model:XLM-R has hidden_size768, and layers:13
Batch: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 370/370 [02:14<00:00,  2.75it/s]
run inference sentence:88860, model:XLM-R has hidden_size768, and layers:13
Batch: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 348/348 [01:53<00:00,  3.06it/s]
saved file to ./dataset/bucc/bucc2017/zh-en/
(ljh) zhangrichong@gpu-15:~/data/ljh/multilingual$ conda activate ljh_faiss
(ljh_faiss) zhangrichong@gpu-15:~/data/ljh/multilingual$ python evaluate_tools.py 

 evaluate on XLM-R, layer-1 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R_zh_1.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R_en_1.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
margin k calculated
[[67488, 4757, 0.9999451637268066], [4971, 9364, 0.9999203681945801], [49092, 48440, 0.9999129772186279], [51867, 43380, 0.9998880624771118], [85680, 39028, 0.9998760223388672]]
optimized bucc f1: 0.0035601522809661787, threshold: 0.9897887110710144, total true positive: 165

 evaluate on XLM-R, layer-5 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R_zh_5.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R_en_5.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
margin k calculated
[[70402, 53878, 0.9999915957450867], [57160, 58636, 0.9999829530715942], [52989, 17639, 0.9999815225601196], [35223, 51186, 0.9999748468399048], [68197, 79563, 0.999971330165863]]
optimized bucc f1: 0.018792669408559264, threshold: 0.9896153211593628, total true positive: 907

 evaluate on XLM-R, layer-7 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R_zh_7.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R_en_7.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
margin k calculated
[[94035, 10535, 0.9999744296073914], [83588, 38576, 0.9999740123748779], [24798, 53633, 0.999967634677887], [86077, 26737, 0.9999662637710571], [41317, 73938, 0.9999660849571228]]
optimized bucc f1: 0.025772768708046737, threshold: 0.9865216016769409, total true positive: 1244

 evaluate on XLM-R, layer-8 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R_zh_8.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R_en_8.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
margin k calculated
[[42518, 20792, 0.9999970197677612], [69198, 10056, 0.99997478723526], [84924, 41638, 0.9999738335609436], [21619, 9362, 0.9999680519104004], [9883, 82393, 0.9999638795852661]]
optimized bucc f1: 0.024840731340964414, threshold: 0.9872369170188904, total true positive: 1199

 evaluate on XLM-R, layer-12 in zh-en setting
load ./dataset/bucc/bucc2017/zh-en/XLM-R_zh_12.npy, ./dataset/bucc/bucc2017/zh-en/XLM-R_en_12.npy
zh-en training gold has: 1899
doc sent_num:88860, query sent_num:94637
candidates generated and similarity generated
margin k calculated
[[13394, 24979, 1.0000000596046448], [23749, 26124, 0.9999974966049194], [78505, 30269, 0.9999973177909851], [12801, 74910, 0.9999972581863403], [66573, 61116, 0.9999967813491821]]
optimized bucc f1: 0.0077549116168161335, threshold: 0.9992965459823608, total true positive: 374

```


```shell
ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ ./bucc.sh 

Processing BUCC data in .
 - extract files ./embed/bucc2018.fr-en.dev in en
 - extract files ./embed/bucc2018.fr-en.dev in fr
 - extract files ./embed/bucc2018.fr-en.train in en
 - extract files ./embed/bucc2018.fr-en.train in fr
 - extract files ./embed/bucc2018.fr-en.test in en
 - extract files ./embed/bucc2018.fr-en.test in fr
2023-08-24 22:25:12,474 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:25:13,077 | INFO | embed | transfer encoder to GPU
2023-08-24 22:25:15,351 | INFO | preprocess | tokenizing  in language fr  
2023-08-24 22:25:33,536 | INFO | preprocess | fastBPE: processing tok
2023-08-24 22:25:35,983 | INFO | embed | encoding /tmp/tmp1ei5o6f5/bpe to ./embed/bucc2018.fr-en.train.enc.fr
2023-08-24 22:25:38,927 | INFO | embed | encoded 10000 sentences
2023-08-24 22:25:41,618 | INFO | embed | encoded 20000 sentences
2023-08-24 22:25:44,252 | INFO | embed | encoded 30000 sentences
2023-08-24 22:25:46,856 | INFO | embed | encoded 40000 sentences
2023-08-24 22:25:49,461 | INFO | embed | encoded 50000 sentences
2023-08-24 22:25:52,058 | INFO | embed | encoded 60000 sentences
2023-08-24 22:25:54,657 | INFO | embed | encoded 70000 sentences
2023-08-24 22:25:57,262 | INFO | embed | encoded 80000 sentences
2023-08-24 22:25:59,884 | INFO | embed | encoded 90000 sentences
2023-08-24 22:26:02,480 | INFO | embed | encoded 100000 sentences
2023-08-24 22:26:05,099 | INFO | embed | encoded 110000 sentences
2023-08-24 22:26:07,794 | INFO | embed | encoded 120000 sentences
2023-08-24 22:26:10,388 | INFO | embed | encoded 130000 sentences
2023-08-24 22:26:13,028 | INFO | embed | encoded 140000 sentences
2023-08-24 22:26:15,632 | INFO | embed | encoded 150000 sentences
2023-08-24 22:26:18,235 | INFO | embed | encoded 160000 sentences
2023-08-24 22:26:20,823 | INFO | embed | encoded 170000 sentences
2023-08-24 22:26:23,415 | INFO | embed | encoded 180000 sentences
2023-08-24 22:26:25,982 | INFO | embed | encoded 190000 sentences
2023-08-24 22:26:28,531 | INFO | embed | encoded 200000 sentences
2023-08-24 22:26:31,098 | INFO | embed | encoded 210000 sentences
2023-08-24 22:26:33,762 | INFO | embed | encoded 220000 sentences
2023-08-24 22:26:36,356 | INFO | embed | encoded 230000 sentences
2023-08-24 22:26:38,984 | INFO | embed | encoded 240000 sentences
2023-08-24 22:26:41,568 | INFO | embed | encoded 250000 sentences
2023-08-24 22:26:44,165 | INFO | embed | encoded 260000 sentences
2023-08-24 22:26:46,780 | INFO | embed | encoded 270000 sentences
2023-08-24 22:26:47,281 | INFO | embed | encoded 271874 sentences in 71s
2023-08-24 22:26:49,896 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:26:50,505 | INFO | embed | transfer encoder to GPU
2023-08-24 22:26:52,815 | INFO | preprocess | tokenizing  in language en  
2023-08-24 22:27:16,287 | INFO | preprocess | fastBPE: processing tok
2023-08-24 22:27:19,237 | INFO | embed | encoding /tmp/tmpe7jrrhea/bpe to ./embed/bucc2018.fr-en.train.enc.en
2023-08-24 22:27:21,983 | INFO | embed | encoded 10000 sentences
2023-08-24 22:27:24,415 | INFO | embed | encoded 20000 sentences
2023-08-24 22:27:26,886 | INFO | embed | encoded 30000 sentences
2023-08-24 22:27:29,315 | INFO | embed | encoded 40000 sentences
2023-08-24 22:27:31,709 | INFO | embed | encoded 50000 sentences
2023-08-24 22:27:34,169 | INFO | embed | encoded 60000 sentences
2023-08-24 22:27:36,596 | INFO | embed | encoded 70000 sentences
2023-08-24 22:27:39,051 | INFO | embed | encoded 80000 sentences
2023-08-24 22:27:41,471 | INFO | embed | encoded 90000 sentences
2023-08-24 22:27:43,887 | INFO | embed | encoded 100000 sentences
2023-08-24 22:27:46,302 | INFO | embed | encoded 110000 sentences
2023-08-24 22:27:48,830 | INFO | embed | encoded 120000 sentences
2023-08-24 22:27:51,308 | INFO | embed | encoded 130000 sentences
2023-08-24 22:27:53,735 | INFO | embed | encoded 140000 sentences
2023-08-24 22:27:56,162 | INFO | embed | encoded 150000 sentences
2023-08-24 22:27:58,601 | INFO | embed | encoded 160000 sentences
2023-08-24 22:28:01,033 | INFO | embed | encoded 170000 sentences
2023-08-24 22:28:03,428 | INFO | embed | encoded 180000 sentences
2023-08-24 22:28:05,834 | INFO | embed | encoded 190000 sentences
2023-08-24 22:28:08,300 | INFO | embed | encoded 200000 sentences
2023-08-24 22:28:10,804 | INFO | embed | encoded 210000 sentences
2023-08-24 22:28:13,426 | INFO | embed | encoded 220000 sentences
2023-08-24 22:28:15,919 | INFO | embed | encoded 230000 sentences
2023-08-24 22:28:18,412 | INFO | embed | encoded 240000 sentences
2023-08-24 22:28:20,938 | INFO | embed | encoded 250000 sentences
2023-08-24 22:28:23,413 | INFO | embed | encoded 260000 sentences
2023-08-24 22:28:25,876 | INFO | embed | encoded 270000 sentences
2023-08-24 22:28:28,387 | INFO | embed | encoded 280000 sentences
2023-08-24 22:28:30,905 | INFO | embed | encoded 290000 sentences
2023-08-24 22:28:33,375 | INFO | embed | encoded 300000 sentences
2023-08-24 22:28:35,873 | INFO | embed | encoded 310000 sentences
2023-08-24 22:28:38,462 | INFO | embed | encoded 320000 sentences
2023-08-24 22:28:40,951 | INFO | embed | encoded 330000 sentences
2023-08-24 22:28:43,427 | INFO | embed | encoded 340000 sentences
2023-08-24 22:28:45,882 | INFO | embed | encoded 350000 sentences
2023-08-24 22:28:48,304 | INFO | embed | encoded 360000 sentences
2023-08-24 22:28:50,693 | INFO | embed | encoded 369810 sentences in 91s
LASER: tool to search, score or mine bitexts
 - knn will run on all available GPUs (recommended)
 - loading texts ./embed/bucc2018.fr-en.train.txt.fr: 271874 lines, 270775 unique
 - loading texts ./embed/bucc2018.fr-en.train.txt.en: 369810 lines, 368033 unique
 - Embeddings: ./embed/bucc2018.fr-en.train.enc.fr, 271874x1024
 - unify embeddings: 271874 -> 270775
 - Embeddings: ./embed/bucc2018.fr-en.train.enc.en, 369810x1024
 - unify embeddings: 369810 -> 368033
 - perform 4-nn source against target
 - perform 4-nn target against source
 - mining for parallel data
 - scoring 270775 candidates
 - scoring 368033 candidates
 - writing alignments to ./embed/bucc2018.fr-en.train.candidates.tsv
2023-08-24 22:29:52,684 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:29:53,304 | INFO | embed | transfer encoder to GPU
2023-08-24 22:29:56,327 | INFO | preprocess | tokenizing  in language fr  
2023-08-24 22:30:14,836 | INFO | preprocess | fastBPE: processing tok
2023-08-24 22:30:17,264 | INFO | embed | encoding /tmp/tmpu3qpv1hy/bpe to ./embed/bucc2018.fr-en.test.enc.fr
2023-08-24 22:30:20,252 | INFO | embed | encoded 10000 sentences
2023-08-24 22:30:22,888 | INFO | embed | encoded 20000 sentences
2023-08-24 22:30:25,519 | INFO | embed | encoded 30000 sentences
2023-08-24 22:30:28,138 | INFO | embed | encoded 40000 sentences
2023-08-24 22:30:30,709 | INFO | embed | encoded 50000 sentences
2023-08-24 22:30:33,291 | INFO | embed | encoded 60000 sentences
2023-08-24 22:30:35,878 | INFO | embed | encoded 70000 sentences
2023-08-24 22:30:38,492 | INFO | embed | encoded 80000 sentences
2023-08-24 22:30:41,092 | INFO | embed | encoded 90000 sentences
2023-08-24 22:30:43,707 | INFO | embed | encoded 100000 sentences
2023-08-24 22:30:46,306 | INFO | embed | encoded 110000 sentences
2023-08-24 22:30:48,983 | INFO | embed | encoded 120000 sentences
2023-08-24 22:30:51,584 | INFO | embed | encoded 130000 sentences
2023-08-24 22:30:54,212 | INFO | embed | encoded 140000 sentences
2023-08-24 22:30:56,802 | INFO | embed | encoded 150000 sentences
2023-08-24 22:30:59,381 | INFO | embed | encoded 160000 sentences
2023-08-24 22:31:01,972 | INFO | embed | encoded 170000 sentences
2023-08-24 22:31:04,597 | INFO | embed | encoded 180000 sentences
2023-08-24 22:31:07,179 | INFO | embed | encoded 190000 sentences
2023-08-24 22:31:09,751 | INFO | embed | encoded 200000 sentences
2023-08-24 22:31:12,363 | INFO | embed | encoded 210000 sentences
2023-08-24 22:31:15,045 | INFO | embed | encoded 220000 sentences
2023-08-24 22:31:17,630 | INFO | embed | encoded 230000 sentences
2023-08-24 22:31:20,207 | INFO | embed | encoded 240000 sentences
2023-08-24 22:31:22,769 | INFO | embed | encoded 250000 sentences
2023-08-24 22:31:25,367 | INFO | embed | encoded 260000 sentences
2023-08-24 22:31:27,927 | INFO | embed | encoded 270000 sentences
2023-08-24 22:31:29,747 | INFO | embed | encoded 276833 sentences in 72s
2023-08-24 22:31:32,240 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:31:32,867 | INFO | embed | transfer encoder to GPU
2023-08-24 22:31:35,461 | INFO | preprocess | tokenizing  in language en  
2023-08-24 22:31:59,157 | INFO | preprocess | fastBPE: processing tok
2023-08-24 22:32:02,220 | INFO | embed | encoding /tmp/tmpk1m2g41_/bpe to ./embed/bucc2018.fr-en.test.enc.en
2023-08-24 22:32:04,938 | INFO | embed | encoded 10000 sentences
2023-08-24 22:32:07,345 | INFO | embed | encoded 20000 sentences
2023-08-24 22:32:09,825 | INFO | embed | encoded 30000 sentences
2023-08-24 22:32:12,260 | INFO | embed | encoded 40000 sentences
2023-08-24 22:32:14,659 | INFO | embed | encoded 50000 sentences
2023-08-24 22:32:17,074 | INFO | embed | encoded 60000 sentences
2023-08-24 22:32:19,554 | INFO | embed | encoded 70000 sentences
2023-08-24 22:32:21,997 | INFO | embed | encoded 80000 sentences
2023-08-24 22:32:24,427 | INFO | embed | encoded 90000 sentences
2023-08-24 22:32:26,845 | INFO | embed | encoded 100000 sentences
2023-08-24 22:32:29,229 | INFO | embed | encoded 110000 sentences
2023-08-24 22:32:31,766 | INFO | embed | encoded 120000 sentences
2023-08-24 22:32:34,219 | INFO | embed | encoded 130000 sentences
2023-08-24 22:32:36,655 | INFO | embed | encoded 140000 sentences
2023-08-24 22:32:39,081 | INFO | embed | encoded 150000 sentences
2023-08-24 22:32:41,510 | INFO | embed | encoded 160000 sentences
2023-08-24 22:32:43,911 | INFO | embed | encoded 170000 sentences
2023-08-24 22:32:46,338 | INFO | embed | encoded 180000 sentences
2023-08-24 22:32:48,737 | INFO | embed | encoded 190000 sentences
2023-08-24 22:32:51,200 | INFO | embed | encoded 200000 sentences
2023-08-24 22:32:53,699 | INFO | embed | encoded 210000 sentences
2023-08-24 22:32:56,269 | INFO | embed | encoded 220000 sentences
2023-08-24 22:32:58,764 | INFO | embed | encoded 230000 sentences
2023-08-24 22:33:01,259 | INFO | embed | encoded 240000 sentences
2023-08-24 22:33:03,755 | INFO | embed | encoded 250000 sentences
2023-08-24 22:33:06,214 | INFO | embed | encoded 260000 sentences
2023-08-24 22:33:08,668 | INFO | embed | encoded 270000 sentences
2023-08-24 22:33:11,156 | INFO | embed | encoded 280000 sentences
2023-08-24 22:33:13,739 | INFO | embed | encoded 290000 sentences
2023-08-24 22:33:16,227 | INFO | embed | encoded 300000 sentences
2023-08-24 22:33:18,686 | INFO | embed | encoded 310000 sentences
2023-08-24 22:33:21,253 | INFO | embed | encoded 320000 sentences
2023-08-24 22:33:23,702 | INFO | embed | encoded 330000 sentences
2023-08-24 22:33:26,178 | INFO | embed | encoded 340000 sentences
2023-08-24 22:33:28,617 | INFO | embed | encoded 350000 sentences
2023-08-24 22:33:31,081 | INFO | embed | encoded 360000 sentences
2023-08-24 22:33:33,524 | INFO | embed | encoded 370000 sentences
2023-08-24 22:33:34,356 | INFO | embed | encoded 373459 sentences in 92s
LASER: tool to search, score or mine bitexts
 - knn will run on all available GPUs (recommended)
 - loading texts ./embed/bucc2018.fr-en.test.txt.fr: 276833 lines, 275724 unique
 - loading texts ./embed/bucc2018.fr-en.test.txt.en: 373459 lines, 371621 unique
 - Embeddings: ./embed/bucc2018.fr-en.test.enc.fr, 276833x1024
 - unify embeddings: 276833 -> 275724
 - Embeddings: ./embed/bucc2018.fr-en.test.enc.en, 373459x1024
 - unify embeddings: 373459 -> 371621
 - perform 4-nn source against target
 - perform 4-nn target against source
 - mining for parallel data
 - scoring 275724 candidates
 - scoring 371621 candidates
 - writing alignments to ./embed/bucc2018.fr-en.test.candidates.tsv
LASER: tools for BUCC bitext mining
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/task/bucc/bucc.py", line 77, in <module>
    and not (args.gold and args.threshold > 0), \
AssertionError: Either "--gold" or "--threshold" must be specified
 - extract files ./embed/bucc2018.de-en.dev in en
 - extract files ./embed/bucc2018.de-en.dev in de
 - extract files ./embed/bucc2018.de-en.train in en
cat: ./bucc2018/de-en/de-en.training.en: No such file or directory
cat: ./bucc2018/de-en/de-en.training.en: No such file or directory
 - extract files ./embed/bucc2018.de-en.train in de
cat: ./bucc2018/de-en/de-en.training.de: No such file or directory
cat: ./bucc2018/de-en/de-en.training.de: No such file or directory
 - extract files ./embed/bucc2018.de-en.test in en
cat: ./bucc2018/de-en/de-en.test.en: No such file or directory
cat: ./bucc2018/de-en/de-en.test.en: No such file or directory
 - extract files ./embed/bucc2018.de-en.test in de
2023-08-24 22:34:33,066 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:34:33,676 | INFO | embed | transfer encoder to GPU
2023-08-24 22:34:36,202 | INFO | preprocess | tokenizing  in language de  
2023-08-24 22:34:36,270 | INFO | preprocess | fastBPE: processing tok
2023-08-24 22:34:36,337 | INFO | embed | encoding /tmp/tmp003eiu31/bpe to ./embed/bucc2018.de-en.train.enc.de
2023-08-24 22:34:36,338 | INFO | embed | encoded 0 sentences in 0s
2023-08-24 22:34:38,755 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:34:39,385 | INFO | embed | transfer encoder to GPU
2023-08-24 22:34:41,845 | INFO | preprocess | tokenizing  in language en  
2023-08-24 22:34:41,913 | INFO | preprocess | fastBPE: processing tok
2023-08-24 22:34:41,980 | INFO | embed | encoding /tmp/tmpzlq9sk_g/bpe to ./embed/bucc2018.de-en.train.enc.en
2023-08-24 22:34:41,981 | INFO | embed | encoded 0 sentences in 0s
LASER: tool to search, score or mine bitexts
 - knn will run on all available GPUs (recommended)
 - loading texts ./embed/bucc2018.de-en.train.txt.de: 0 lines, 0 unique
 - loading texts ./embed/bucc2018.de-en.train.txt.en: 0 lines, 0 unique
 - Embeddings: ./embed/bucc2018.de-en.train.enc.de, 0x1024
 - unify embeddings: 0 -> 0
 - Embeddings: ./embed/bucc2018.de-en.train.enc.en, 0x1024
 - unify embeddings: 0 -> 0
 - perform 4-nn source against target
/home/LAB/niezj/multilingual/source/mine_bitexts.py:231: RuntimeWarning: Mean of empty slice.
  x2y_mean = x2y_sim.mean(axis=1)
 - perform 4-nn target against source
/home/LAB/niezj/multilingual/source/mine_bitexts.py:237: RuntimeWarning: Mean of empty slice.
  y2x_mean = y2x_sim.mean(axis=1)
 - mining for parallel data
 - scoring 0 candidates
 - scoring 0 candidates
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/source/mine_bitexts.py", line 273, in <module>
    fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
ValueError: attempt to get argmax of an empty sequence
LASER: tools for BUCC bitext mining
 - reading sentences and IDs
 - reading candidates ./embed/bucc2018.de-en.train.candidates.tsv
 - optimizing threshold on gold alignments ./bucc2018/de-en/de-en.training.gold
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/task/bucc/bucc.py", line 132, in <module>
    gold = {line.strip() for line in open(args.gold)}
FileNotFoundError: [Errno 2] No such file or directory: './bucc2018/de-en/de-en.training.gold'
2023-08-24 22:34:46,896 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:34:47,504 | INFO | embed | transfer encoder to GPU
2023-08-24 22:34:49,955 | INFO | preprocess | tokenizing  in language de  
2023-08-24 22:35:15,008 | INFO | preprocess | fastBPE: processing tok
2023-08-24 22:35:21,741 | INFO | embed | encoding /tmp/tmpx9mij5qn/bpe to ./embed/bucc2018.de-en.test.enc.de
2023-08-24 22:35:24,903 | INFO | embed | encoded 10000 sentences
2023-08-24 22:35:27,702 | INFO | embed | encoded 20000 sentences
2023-08-24 22:35:30,500 | INFO | embed | encoded 30000 sentences
2023-08-24 22:35:33,393 | INFO | embed | encoded 40000 sentences
2023-08-24 22:35:36,255 | INFO | embed | encoded 50000 sentences
^CTraceback (most recent call last):
  File "/home/LAB/niezj/multilingual/source/embed.py", line 615, in <module>
    embed_sentences(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 545, in embed_sentences
    EncodeFile(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 440, in EncodeFile
    EncodeFilep(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 403, in EncodeFilep
    encoded = encoder.encode_sentences(sentences)
  File "/home/LAB/niezj/multilingual/source/embed.py", line 182, in encode_sentences
    results.append(self._process_batch(batch))
  File "/home/LAB/niezj/multilingual/source/embed.py", line 121, in _process_batch
    sentemb = self.encoder(tokens, lengths)["sentemb"]
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/LAB/niezj/multilingual/source/embed.py", line 336, in forward
    x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)
KeyboardInterrupt

(ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ ./bucc.sh 

Processing BUCC data in .
LASER: tools for BUCC bitext mining
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/task/bucc/bucc.py", line 77, in <module>
    and not (args.gold and args.threshold > 0), \
AssertionError: Either "--gold" or "--threshold" must be specified
2023-08-24 22:40:05,497 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:40:06,115 | INFO | embed | transfer encoder to GPU
^C
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/source/embed.py", line 615, in <module>
    embed_sentences(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 494, in embed_sentences
    encoder = load_model(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 367, in load_model
    return SentenceEncoder(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 109, in __init__
    self.encoder.cuda()
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 905, in cuda
    return self._apply(lambda t: t.cuda(device))
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 820, in _apply
    param_applied = fn(param)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 905, in <lambda>
    return self._apply(lambda t: t.cuda(device))
KeyboardInterrupt

(ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ 
(ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ ./bucc.sh 

Processing BUCC data in .
optimize threshold on BUCC training data and provided gold alignments
extract test bitexts for treshhold optimized on train
LASER: tools for BUCC bitext mining
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/task/bucc/bucc.py", line 77, in <module>
    and not (args.gold and args.threshold > 0), \
AssertionError: Either "--gold" or "--threshold" must be specified
2023-08-24 22:47:23,754 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:47:24,358 | INFO | embed | transfer encoder to GPU
^CTraceback (most recent call last):
  File "/home/LAB/niezj/multilingual/source/embed.py", line 615, in <module>
    embed_sentences(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 494, in embed_sentences
    encoder = load_model(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 367, in load_model
    return SentenceEncoder(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 109, in __init__
    self.encoder.cuda()
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 905, in cuda
    return self._apply(lambda t: t.cuda(device))
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 820, in _apply
    param_applied = fn(param)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 905, in <lambda>
    return self._apply(lambda t: t.cuda(device))
KeyboardInterrupt

(ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ ./bucc.sh 

Processing BUCC data in .
optimize threshold on BUCC training data and provided gold alignments
extract test bitexts for treshhold optimized on train
LASER: tools for BUCC bitext mining
None 0.0
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/task/bucc/bucc.py", line 78, in <module>
    and not (args.gold and args.threshold > 0), \
AssertionError: Either "--gold" or "--threshold" must be specified
2023-08-24 22:49:38,997 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
^CTraceback (most recent call last):
  File "/home/LAB/niezj/multilingual/source/embed.py", line 615, in <module>
    embed_sentences(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 494, in embed_sentences
    encoder = load_model(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 367, in load_model
    return SentenceEncoder(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 88, in __init__
    self.encoder = LaserLstmEncoder(**state_dict["params"])
  File "/home/LAB/niezj/multilingual/source/embed.py", line 275, in __init__
    self.lstm = nn.LSTM(
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 711, in __init__
    super().__init__('LSTM', *args, **kwargs)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 132, in __init__
    self.reset_parameters()
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 209, in reset_parameters
    init.uniform_(weight, -stdv, stdv)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/init.py", line 137, in uniform_
    return _no_grad_uniform_(tensor, a, b)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/init.py", line 14, in _no_grad_uniform_
    return tensor.uniform_(a, b)
KeyboardInterrupt

(ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ ./bucc.sh 

Processing BUCC data in .
optimize threshold on BUCC training data and provided gold alignments
./bucc2018/fr-en/fr-en.training.gold
extract test bitexts for treshhold optimized on train
LASER: tools for BUCC bitext mining
None 0.0
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/task/bucc/bucc.py", line 78, in <module>
    and not (args.gold and args.threshold > 0), \
AssertionError: Either "--gold" or "--threshold" must be specified
2023-08-24 22:50:22,274 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
^CTraceback (most recent call last):
  File "/home/LAB/niezj/multilingual/source/embed.py", line 615, in <module>
    embed_sentences(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 494, in embed_sentences
    encoder = load_model(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 367, in load_model
    return SentenceEncoder(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 88, in __init__
    self.encoder = LaserLstmEncoder(**state_dict["params"])
  File "/home/LAB/niezj/multilingual/source/embed.py", line 271, in __init__
    self.embed_tokens = nn.Embedding(
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/sparse.py", line 144, in __init__
    self.reset_parameters()
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/sparse.py", line 153, in reset_parameters
    init.normal_(self.weight)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/init.py", line 155, in normal_
    return _no_grad_normal_(tensor, mean, std)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/init.py", line 19, in _no_grad_normal_
    return tensor.normal_(mean, std)
KeyboardInterrupt

(ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ vim bucc.sh
(ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ ./bucc.sh 

Processing BUCC data in .
optimize threshold on BUCC training data and provided gold alignments
./bucc2018/fr-en/fr-en.training.gold
extract test bitexts for treshhold optimized on train
LASER: tools for BUCC bitext mining
None 0.0
Traceback (most recent call last):
  File "/home/LAB/niezj/multilingual/task/bucc/bucc.py", line 78, in <module>
    and not (args.gold and args.threshold > 0), \
AssertionError: Either "--gold" or "--threshold" must be specified
2023-08-24 22:51:46,955 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
^CTraceback (most recent call last):
  File "/home/LAB/niezj/multilingual/source/embed.py", line 615, in <module>
    embed_sentences(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 494, in embed_sentences
    encoder = load_model(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 367, in load_model
    return SentenceEncoder(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 86, in __init__
    state_dict = torch.load(model_path)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/serialization.py", line 815, in load
    return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/serialization.py", line 1051, in _legacy_load
    typed_storage._untyped_storage._set_from_file(
KeyboardInterrupt

(ljh_new) niezj@dell-gpu-06:~/multilingual/task/bucc$ ./bucc.sh 

Processing BUCC data in .
optimize threshold on BUCC training data and provided gold alignments
./bucc2018/fr-en/fr-en.training.gold
LASER: tools for BUCC bitext mining
./bucc2018/fr-en/fr-en.training.gold -1
 - reading sentences and IDs
 - reading candidates ./embed/bucc2018.fr-en.train.candidates.tsv
 - optimizing threshold on gold alignments ./bucc2018/fr-en/fr-en.training.gold
 - best threshold=1.088131: precision=91.52, recall=93.32, F1=92.41
extract test bitexts for treshhold optimized on train
LASER: tools for BUCC bitext mining
None 1.088131
 - reading sentences and IDs
 - reading candidates ./embed/bucc2018.fr-en.test.candidates.tsv
 - extracting bitexts for threshold 1.088131 into ./embed/bucc2018.fr-en.test.extracted.tsv
2023-08-24 22:53:01,313 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:53:01,929 | INFO | embed | transfer encoder to GPU
2023-08-24 22:53:04,235 | INFO | preprocess | tokenizing  in language de  
2023-08-24 22:53:04,295 | INFO | preprocess | fastBPE: processing tok
2023-08-24 22:53:04,366 | INFO | embed | encoder: bucc2018.de-en.train.enc.de exists already
2023-08-24 22:53:06,743 | INFO | embed | loading encoder: /home/LAB/niezj/multilingual/models/bilstm.93langs.2018-12-26.pt
2023-08-24 22:53:07,344 | INFO | embed | transfer encoder to GPU
^CTraceback (most recent call last):
  File "/home/LAB/niezj/multilingual/source/embed.py", line 615, in <module>
    embed_sentences(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 494, in embed_sentences
    encoder = load_model(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 367, in load_model
    return SentenceEncoder(
  File "/home/LAB/niezj/multilingual/source/embed.py", line 109, in __init__
    self.encoder.cuda()
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 905, in cuda
    return self._apply(lambda t: t.cuda(device))
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 820, in _apply
    param_applied = fn(param)
  File "/home/LAB/niezj/.conda/envs/ljh_new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 905, in <lambda>
    return self._apply(lambda t: t.cuda(device))
KeyboardInterrupt

```