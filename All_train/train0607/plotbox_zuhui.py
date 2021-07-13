import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n_n= [49.34567548037539, 41.83339304124664, 4.918442147036162, 48.0734532597115, 45.6467690461015, 2.5890377501664226, 46.034564369264295, -7.443989019801529, 49.4456754803754, 49.26234214704208, 44.095690474127906, 49.617897702597624, 49.5067865914865, 49.395675480375395, 48.71234214704209, 48.389209075534, 13.072639173986504, 48.956786591486534, 47.562342147042074, 40.98291229678725, 49.466518773285884, 47.245675480375404, 43.24047856180265, 49.39567548037539, 48.82345325815318, 41.479126439865375, 49.23456436926429, 48.97345325815332, 45.69011992485483, 41.74679346763348, 47.68374367860193, 19.815271477899337, 45.762342147042055, 28.01914489009761, 49.062342147042074, 49.35678659148651, 41.27345555037985, 46.88574827703974, 48.667897702632615, 10.235786591486502, 22.571366570780658, 46.95123103593203, 49.2456754803754, 49.40678659148651, 33.25930131836943, 48.510228497194774, 46.51234214704207, 49.07345325815318, 48.98456436929928, -7.95043954040785]
n_na = [49.18456436926428, 47.732922916774, 48.91241774457949, 11.35882882614013, 49.223453576624465, 45.55690422478347, 4.531718655978834, -8.353393996307712, 49.31789770259762, 49.062342147042045, 39.22354619839242, 48.55686218902392, 46.84038809622415, 49.056786591486514, 48.93456436926431, 49.517897702597615, 48.95678659148654, 4.520297928438868, 49.81789770259762, 49.567897702597605, -7.232023229007695, 49.051231035930954, 48.873453258153184, 21.165056150322176, 17.149614427044632, -5.458498133140029, 1.9937340351895347, 49.06789770259763, -6.790160216579798, 49.334564369264285, 45.112459780339016, 4.335786591484526, -4.705361293004918, 37.34893694412714, 46.80104620616274, 30.167988202450903, -8.353393996307712, -3.7964844090856884, 48.67900881370872, 45.30812106056022, 48.334564369264285, 49.256786591486524, 30.987191818201104, 49.56789770259762, 49.667897702597614, 47.73347843385549, 47.804322241592736, 48.62900881534269, 49.20678659148652, 3.368766265097072]
n_a = [49.606786591486504, 34.913563682344005, 44.728194230747775, 48.912342147042075, 46.6401199251278, 49.71789770259762, 39.51707979355096, 49.51789770259762, 46.39145033442875, 43.61789770286297, 49.16789770259764, 48.22335272694032, 47.4178977042452, 49.03318543995255, 47.37900881370875, 48.912342147042054, 40.752589924354936, 48.54557494916254, 29.643049255691658, -4.813801279300707, -7.745247991572179, 49.101231035930944, 49.406786591486515, 48.7615210766082, 48.79013479451183, 48.66349427415279, -6.41479451394331, 47.119080860912085, 48.80123103593094, 48.49320686822995, 48.74011992482094, 46.951231035930945, 49.367897702597624, 0.7509690836014382, 48.65123103593097, 49.41789770259762, -7.068901000248676, 49.317897702597634, 45.58874099736643, 46.184564369264294, 27.58193814458054, 49.41789770259762, 43.07316785253636, -5.452755524980401, 49.2845643692643, 49.35678659148651, -7.0716570016970355, -7.7020973045568875, 49.51789770259762, 45.59323630086091]

# keneng budui
a_a = [43.40241420731663, 47.95123103593098, 11.643049255691649, -8.08126493936574, 10.572371083516613, -8.181612726743134, 48.80540766217479, 48.68456436926432, 46.24011998456175, 48.25123103593096, 13.115271477913867, 31.531938141686673, 48.22900918361487, 48.43456436926431, 48.106786592864225, -8.08126493936574, 48.87345325815318, 48.75678659148653, 25.754160366800797, -4.603806939057151, 48.27900913217999, 49.05678659148652, 28.13253457906322, 39.58627715619602, 46.72345325815317, 7.6771214409316055, 48.71789770259764, 49.24857370852034, 48.967897702597625, 49.5067865914865, 6.665354333633532, 24.826382589024966, 1.1350781555490101, 48.85123103593093, 49.256786591486524, 48.46224161582922, 49.30678659148649, 0.604899217311905, 48.14011992481987, 49.11789770259762, 49.156786591486494, -8.181612726743134, 48.34013213424421, 46.70814535541748, 25.03456436926429, 49.045675480375394, -4.089766571414155, 48.835120256251955, 48.92345325815319, 47.01789770259763]
a_n = [20.375122085889362, 4.95113068757969, 6.387825832696601, 48.579008813708725, -8.492815269127838, 49.417897702597614, -7.808107750239588, -2.178528586240448, 10.270911815639657, 48.72345325815422, 13.627121440931605, 49.206786591486505, 4.603638589303456, 28.026382588557624, 48.94576328733717, 48.20123103748927, -5.862799984446431, -7.600938115627012, 48.9456754803754, 46.00678659148653, 47.229009254862625, 49.00678659148652, 10.384566659625753, 49.07345325815318, 20.814400653986503, 22.781938144505062, 33.36527147791354, 40.53845565123112, 48.983729117515864, 17.8628865914865, 48.68456436951597, 48.79567548037541, 48.01789770259764, 48.684564369264294, 18.64011998456193, 39.621826111050154, 47.9835529887656, 48.71234214704205, 48.67345325815319, 37.11496526843289, 47.928097964422896, 48.251308923829875, 18.321984858094112, 42.749690187245406, 48.42818811318102, 11.289037750183274, 12.833482194627718, 24.990471856763804, -6.943794504914382, 29.435432931764275]
a_d =[5.85415956894102, 13.861528029898174, 49.09567548037539, 49.095675480375405, -4.020504318619462, 49.02345325815334, 31.444371571966048, 48.70123103593096, -2.572607818182016, 4.554235604717288, -5.801647143899281, -8.492815269127838, 48.66234214704207, 48.045675480375394, 48.98456436926426, 2.8160053522466892, 3.0652714779015913, -5.801647143899281, 49.06789770259761, 44.87345325815444, 5.767897702597617, 48.72345325815331, 48.52263021582912, -4.243210866903447, 48.9456754803754, 45.10512230127027, 43.786835015004186, 49.01234214704205, 47.132922986165774, 49.05678659148651, 49.32345325836985, 48.63456436926431, 5.451130504754008, -5.415332086794456, 48.7845643692643, 49.10678659148651, 47.751231035930985, 8.206786591486502, 48.92345325815334, 48.734564369264284, 49.00678659148652, 1.2622416158743093, 48.912342155354736, 48.36696191963637, -1.8451484965112357, 49.23456436926428, 20.48348219462772, 49.606786591486504, 22.561191423266457, -0.7982772207161251]



d_n = [48.54567548038195, 45.89812328671555, 49.1456754803754, 37.050968645343346, 49.36789770259761, -7.5815105531270115, 5.754235604717534, 49.03456442069918, 48.59103115119621, -8.021624677245816, 27.060454774264947, 48.99567548037541, 48.90678659148651, 48.217049726598134, 21.064545489540635, 49.051598584744, 48.984564369264305, 48.60123103593661, 49.206786591486505, 49.14567548037542, -7.476120165927187, 47.99139832291874, 49.19567548037541, 49.06234214704206, 49.10587574220068, 48.7845643692643, 48.7512432453553, 9.74304925569164, 49.50678659148652, 47.69113542983399, 45.349234366958036, -6.8054143029509575, 3.2456754803793864, 48.73456436926429, 48.6234532581532, 48.50123103593096, 47.83838800510014, 1.542889742170999, -7.717903854842202, 49.417897702597614, 47.6327177370172, 11.872371083516299, 7.326382589024975, 49.61789770259761, 48.95678659148653, 6.21283389117456, 48.119227415522076, 34.377965525752906, 49.46789770259762, 48.756786591486524]
d_a = [46.10579256306727, 48.29011992481988, 47.95493995923942, 47.63372918015773, 48.92345325815319, 42.60297019717498, 49.23456436926429, 6.096194602597604, 49.21789770259763, -7.588243204877987, -7.745247991572179, 49.094764631089575, 25.5271214409316, 49.41789770259763, 48.24567548037668, 49.06789770259764, 48.740019393606985, 49.367897702597624, 49.35678659148651, 42.27355830494693, 48.067897702597634, 20.703997308200357, 49.23456436926428, 49.41789770259762, 48.92345325815318, 49.61789770259762, 42.30524383891091, 48.240678102168964, 11.393788107373993, 47.32900881370873, 48.51690082188475, 49.467897702597625, 49.00678659148652, 49.09567548037541, 25.40014281658906, 49.11789770259763, 49.36789770259762, 25.454160366802753, 48.52900881370872, 47.715708455212926, -6.611971856559485, 24.10250943402982, 49.25678659148652, -6.225246502056292, 47.58574325857837, 48.912912215344264, 48.09894125889171, 48.94484022862699, 49.2567865914865, 25.89689770259761]
d_d = [49.19567777073687, 48.86234214704879, 49.66789770259761, 48.9802872118076, 21.77346943239067, 49.3567885633767, 49.20678659148653, 22.630231035463577, 48.94476463108958, 49.21234214704206, 49.0845643692643, 49.20678659148651, 20.704454586016954, -7.626560762227913, 49.195675480375414, 48.8735288556906, 45.395675480376696, 13.46125997231783, 49.1345643692643, 36.490988487319854, 49.13456436926428, 24.511259972405504, 49.556786591486514, 45.328202755156944, 49.323453258153165, 48.51789770259763, 48.52964227012402, 20.16527147791387, 49.09567548037539, 14.528908465357489, 49.39567548037539, 48.812342147042074, 49.1456754803754, 48.39557494916255, 48.680287211807624, 48.74011992481984, 49.44567548037541, 1.4972424126012829, -6.988263206319345, 49.50678659148652, 48.91234214704208, 49.17345325819383, -8.361788382881786, 48.47170715711894, 47.944384735115975, 15.18745721634803, 4.154851509983207, 49.36651877328589, 49.212342147042065, 49.41789770259763]


m_n =[49.66789770259763, 47.69011992481984, -5.805407923107042, 19.60441472780991, 3.4263825890248825, 49.50678659148649, 47.66396381124213, 49.345675480375405, 48.09300580580424, 49.01234214704207, 49.567897702597605, 26.054160366802755, 49.07345325815318, 49.317897702597605, -8.426732108240465, 48.27716218146163, 49.334564369264285, 48.24382884812831, 49.17345325815318, 6.927121440931604, 7.944593305738829, 49.76789770259761, 43.32427490689378, 47.68189469469313, 49.1845643692643, 49.71789770259761, 10.09066436926429, 49.26789770259761, 49.11362054514095, 49.11234214704206, -7.745247991572179, 49.26789770259762, 49.51789770259762, 48.97354303700514, 10.9179882024509, 48.86224161582922, 49.3345643692643, 19.965271477913866, 49.56789770259762, 49.56789770259762, -2.3832133994910043, 49.606786591486504, 18.115108419311465, 49.28456436926429, -8.426732108240465, 45.084564369264484, 30.196897702597607, 49.817897702597605, 30.039032541404964, 49.34567548037539]

m_a = [1.9882325520412891, 49.61789770259762, 49.41789770259762, 49.71789770259761, 48.42900881370874, 17.643783147042058, 49.417897702597614, 49.1345643692643, 49.46789770259761, 49.2345643692643, 49.50678659148652, 49.18457855057884, 49.46789770259761, 49.417897702597614, 47.84382890787022, 10.484564369264294, 49.56789770259762, 49.26789770259762, 49.25678659148652, 49.40678659148652, 49.45678659148652, 20.10387010423839, 49.40678659148651, 49.45678659148651, 37.58823255204271, 45.44011998486208, 22.358387702597607, -2.64161231875154, 29.14844175264481, 49.367897702597624, 49.31789770259762, 49.31789770259761, 49.184564369264294, 48.8234532584114, -3.0821022053894493, 49.2956754803754, 48.923453258153195, 49.0845643692643, -8.442815269127838, 49.134564369264304, 48.85021965543226, 49.4345643692643, 48.86133076654338, 49.2345643692643, -8.131612726743134, 49.24567579884668, 15.326382589024972, 49.61789770259762, 47.2234532581532, 49.567897702597605]

m_d = [48.995675480375404, 49.51789770259762, 46.15803070981634, 49.323453258153165, 45.18298916821783, 21.127706657491224, 14.00416036671509, 49.86789770259762, 49.76789770259761, 19.258387702597606, 49.86789770259762, 11.560048055238235, 49.61789770259762, 45.13386535807307, 49.14057528059464, 48.962342147042065, 45.10037599589232, 49.556786591486514, 47.24774029092593, 49.23456436926429, 49.567897702597605, 49.4956754803754, 49.56789770259762, 48.66068385296958, -7.242860671942431, -2.702723503755431, -2.2721022953799315, 49.295675480375394, 49.27335272694031, 1.8137878845826023, -3.5445919273410107, 46.316362939932034, 47.893005805804236, 46.001348669225976, 49.42335272694031, 48.79011992481984, 49.667897702597614, 13.90528410301911, 49.18456436927099, 49.71789770259761, -6.414247352077913, -8.37580991502304, 47.49856370315615, 49.434564369264294, 49.3956754803754, 48.3345643692643, 49.71789770259761, -1.703805403896947, 49.3456754803754, -8.492815269127838]


data_n = {
    'n_n': n_n,
    'n_a': n_a,
    'n_d': n_na
}

data_a = {
    'a_n': a_n,
    'a_a': a_a,
    'a_d': a_d
}
data_d = {
    'd_n': d_n,
    'd_a': d_a,
    'd_d': d_d
}

data_m = {
    'm_n': m_n,
    'm_a': m_a,
    'm_d': m_d
}



#
# test = {
#     'a':
#     'b':
# }

df = pd.DataFrame(data_n)
df2 = pd.DataFrame(data_a)
df3 = pd.DataFrame(data_d)
df4 = pd.DataFrame(data_m)

df.plot.box(title="n")
df2.plot.box(title="a")
df3.plot.box(title="d")
df4.plot.box(title="m")
# df.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', density = True, label = '直方图')
# df.plot(kind = 'kde',  color = 'steelblue',  label = '直方图')
# df.plot(kind = 'kde',   label = '直方图')
# df.plot.box(title="hua tu")
# df2.plot.box(title="hua tu")
# df3.plot.box(title="hua tu")
plt.grid(linestyle="--", alpha=0.3)
plt.show()




