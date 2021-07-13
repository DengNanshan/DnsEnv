import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = [0.6736043127382718, -7.532539237674751, 45.97072590646556, -4.946755686610429, -3.891811810257094, -0.7748694149836393, 43.85961479535443, 41.08183701757666, -0.16207805429513833, -3.3882036410679275, 3.3909999165282567, 6.128913054761194, 41.04294812868777, -2.758160619054922, -1.2833074145473464, 42.78739257313223, -9.130307670574364, -6.339789724565038, 43.365170350909985, -1.8363831617203266, 39.720725906465546, -1.5856619513601, 45.41517035090998, 20.624014645707916, 42.870725906465566, 39.54850368424333, -1.708325699115532, -3.988841548459254, -0.7965273188111031, -1.8974786830023813, 40.587392573132206, 1.7069356937599736, -5.127212610488318, 40.04294812868776, 2.4677603565330237, 43.85961479535443, -4.090157038417163, -2.7344299234632423, -6.218992641584624, 0.5394387294918008, -0.49024418813210957, 6.859144878938993, -2.0734405167574943, -8.18870596866041, 40.64850368424332, 44.51517035090999, 3.696339102914788, 43.70961479535444, -3.470778469754408, 41.95961479535443, -2.7462925000620544, -8.735312215963871, 1.490555719631864, 0.06538943054043678, 41.315170350909995, 43.46517035090999, -1.7819376269554255, 18.042758788556917, 2.7737295186662045, 46.25405923979888, 17.23246842940841, 3.517785035576452, -2.157212260891283, 41.44850368424332, -8.40195911621054, 37.820725906465555, 4.788888133431085, 0.7855489853823663, 5.801723770038327, -0.06643540900501677, -2.886133469164996, -1.605073610472112, 42.75961479535442, -2.9569121591548315, 40.4262814620211, 40.41517035090999, 45.192948128687775, 6.128161911175141, 43.242948128687765, -2.4096337475024914, -1.0437837456319787, 0.6076021381313055, -4.915194135250682, 40.11517035090999, 42.070725906465555, 3.091584149965909, 1.9062414560224994, 43.90961479535443, 39.88183701757666, 0.18375457796541816, 32.60961479539621, 1.8032320160249817, 42.58739257313221, 40.69850368424332, 34.31517035094502, 32.00405924009922, 7.44361931261259, 1.6579964636503508, 46.354059239798886, -4.004327375079338]
c = [48.84777142023128, 45.30332698548417, 48.65888253134239, 13.411985604574731, 48.36443808689794, 48.60888253134239, 6.644108583361792, 48.608882531342395, 48.80888253134239, 48.11443808690464, 48.0422158646757, 48.15332697578683, 48.47554919800906, 47.99221586467569, 11.880022578927976, 49.05888253134239, 48.225549198009055, 48.21443808689794, 47.32554919800906, 48.56443808689794, 28.580022578928038, 48.3588825313424, 48.10332697578684, 48.37554919800905, -6.270875323940406, 7.429217380787474, 48.60888253134239, 48.49777142023128, 48.62554919800905, 48.5088825313424, 7.245884047454137, 48.192215864675724, 48.30332697578682, 49.208882531342375, 48.003326975786834, 48.32554919800906, 48.43666030912018, 3.58403408443362, 18.617367417769742, 48.508882531342394, 48.90888253134238, 48.78666030912017, 48.57554919800904, 48.70888253134238, 45.97818224270781, 14.2340340844364, 48.14221586467572, 47.52554963915641, 44.647771420231265, 48.85888253134239, 48.59777142023128, 47.60888253134238, 48.47554919800905, 2.6792173807805497, 45.43666030916301, 48.27554919800906, 13.449372531342373, 48.26443808690357, 47.825549198009064, 45.69788909458681, 48.286674490434706, 48.07554919800906, 48.07554919800905, 46.658047279594, 48.08110475356462, 47.35888253134239, -1.5477552242001602, -6.183229262938094, 48.6588825313424, 5.796458436182357, 9.917367417769741, -3.6911158465411456, 47.55030247665371, 48.22554919800905, 48.42554919800905, 47.65250622382424, 3.6545491980062543, 44.44777142023128, 48.75888253134239, 9.09588404745414, 48.74777142023126, 47.795672270169824, 48.17554919800905, 48.32554919800904, 48.45888253134239, 14.61736741776974, 47.291380612927334, 47.81443808689795, 13.703463458689077, 26.835708334825433, 41.37563929533249, 48.52554919800906, 48.397771420231294, 48.758882531342394, 48.85888253134239, 47.4588825313424, 47.63666030912017, 48.153326975786825, 48.17554919800905, -5.164427849824074]
b = [4.001758025569199, 45.05369310779285, 45.181470885570675, 45.37591533002072, 45.32591533001633, 45.70924866334842, 46.575915330015064, 44.7592486633484, 46.53702644112619, 46.31480421890396, 46.97035977445951, 15.039568623480069, 22.92232300982623, 45.598137552237304, -6.421074760891705, 45.10369310779285, 46.970359774459524, 46.17591533001507, 13.740099789754757, 47.51480421890396, 46.13147088557063, -2.393572979453265, 3.294972418227932, 47.131470885570636, 46.63702644112618, 46.637026441126196, 7.102581996906066, 46.36480421890396, 44.03147088557061, 14.848147667274802, 22.332129056114383, 46.8092486633484, 46.51480421890396, -6.475593009365043, -8.467086818547038, 4.4331961559865025, 2.240524423845537, 7.372995478468074, 46.23147088557063, 46.47591533001507, -1.8494641180176847, 45.870359774459516, 46.748137552237296, 46.68702644112618, -5.619164109898284, -4.243979340760491, 46.970359774459524, 2.6648943261422886, 46.63702644112618, 43.47591533001507, 46.8981375522373, -0.4355018900245895, 46.103693107792864, 4.878554398320656, 47.78702644112619, 45.97035977445951, 43.598137552237304, 47.748137552237296, 45.17035977445966, -3.0435827374019624, 42.092581996681766, 46.62591533001506, 45.61480421890395, 47.05924866334841, 46.8981375522373, 45.970359774459524, 45.19258199668175, 34.31490886774827, 47.587026441126184, 47.142581996681756, 37.25497150745663, 46.74813755223731, -3.7191157178316, -0.4528903078940374, 47.68702644112618, 46.25369310779376, 46.05924866334841, 11.475915546431965, 46.99813755223731, 44.720359774459524, 46.52591533001508, 46.13147088557062, 44.970359774459524, 46.759248663348416, 47.253693107792856, 46.08147088557064, 45.53702644112618, 48.242581996681736, 44.38702644112619, 47.23147088557063, 45.29258199668173, 46.41480421890395, 46.98147088557063, 46.242581996681736, 0.5023165320909477, 37.5592487343456, -0.06552509403172557, 46.73702644112618, 45.3981375522373, 44.41480421890395]


data = {
'China': [0,1,22,3,4,5,6,7],
'a': [0.6736043127382718, -7.532539237674751, 45.97072590646556, -4.946755686610429, -3.891811810257094, -0.7748694149836393, 43.85961479535443, 41.08183701757666, -0.16207805429513833, -3.3882036410679275, 3.3909999165282567, 6.128913054761194, 41.04294812868777, -2.758160619054922, -1.2833074145473464, 42.78739257313223, -9.130307670574364, -6.339789724565038, 43.365170350909985, -1.8363831617203266, 39.720725906465546, -1.5856619513601, 45.41517035090998, 20.624014645707916, 42.870725906465566, 39.54850368424333, -1.708325699115532, -3.988841548459254, -0.7965273188111031, -1.8974786830023813, 40.587392573132206, 1.7069356937599736, -5.127212610488318]
}

data_0624_2_IDM_DQN1 = {
    'a': a,
    'b': b,
    'c': c
}

d = [16.348417668134186, -1.2578787643126432, -7.9469278052080226, -1.0478770674159144, 0.7656138886305399, -0.2610933245117, 2.1598864482172218, -8.328819178702547, 24.82081644685221, 47.465170350909986, 47.23183701757665, -1.0856552063061926, 29.99294813833643, 43.192948128687775, 7.8352867218073605, -2.531384962958004, 3.7002312812020453, -7.550834975630023, -2.312862439355235, -2.74784265027016, -0.060357721275114073, -3.987153022968653, 41.5873925731322, -3.510858119451286, 45.476281462021106, 17.22670572410476, 0.6167824046707171, -1.9699025511738375, -3.2270964532255633, 43.41517035090999, 42.59850368424332, 43.476281462021106, 3.4614890634654074, 2.27788201089057, 17.644608213654614, 8.006586092623623, 5.4167458379840685, 0.4911044194275114, 2.7492490380448515, -2.1551077928598623, 37.44850368424332, -1.0279125450879523, 42.59850368424331, -8.482075852343527, -3.5031349040850657, 1.7781650751363252, 0.8913256454570515, -3.3209967503737863, 5.608738964259048, 37.426281462021095, 13.357679520373647, 46.24294812868777, 6.434050224816261, 0.41853930361300407, 12.029742290858964, -6.1165926505320645, 42.63739257313221, 3.3975216928501606, 44.58739257313221, -4.019127004551271, 9.509849327355603, -3.1147071007247398, 3.1857750132619316, -7.156556680959602, 1.6819616493070733, 15.166813143240446, 4.861312414075472, 37.60961479535444, -6.901807556295111, 1.7498619948458387, -1.5436694726176539, -8.315508542650262, -1.2222343510533236, 3.0445491490112246, -0.7698731123861933, 20.16697240555595, 18.444605923300976, -9.002977011321189, 2.032218890112965, -0.4886534694027098, -0.5930233689727782, -4.669478464233035, -5.834117353986167, -3.64456772647143, -3.8440249543935794, 2.4877246674691325, 4.30878994521791, 0.5299966867202617, 5.068886316398665, -1.6624946323374097, 13.21045380989545, 5.044971155861278, 29.25405968285333, -3.978276925137096, -2.3233366764413965, 40.437392573132215, 0.7112119683601357, -7.409219566905742, 46.08183701757666, 45.47628146202112]
e = [46.34258199668173, 45.18147088578732, 5.823270909334466, 38.69813761332084, 43.75924866334861, 45.92035977445951, 40.5981375522373, 3.2583226365804823, 6.331962609964528, 37.99814034323089, -0.3736491335632319, 45.57591533001507, 44.242581996681736, 38.07035977605389, 2.787369379199431, 45.2814708855772, 46.57591533001509, 5.020786007781359, -3.682324676297081, 44.89813755223729, 40.809248664730674, 3.565194163944991, 4.470889083504476, 43.82591533001508, 47.77591533001507, 46.120359774459516, 34.89386084775107, 14.572402009149414, -1.1388023596541021, 44.609248663349476, 47.41480421890394, 8.142480647369045, -2.6888393665144417, 36.759263223183034, -4.7761970078401985, 46.05924866334841, 5.505261456495985, 46.464804218903964, 23.075814981642218, 46.67591533001506, -1.0662347356420092, 6.550075744802813, 45.52591533001598, 47.848137552237304, 3.0501797950634497, 45.19258199668174, 25.593012597068196, 47.34258199668174, 45.90924866334841, 45.77591533001508, 2.4619626668086347, 35.66546952909147, 41.30370531721723, 45.80924866334841, 47.34258199668173, 15.950776620101985, 44.6481375522373, 39.242581996682794, -5.16232829485022, 37.659248726541215, 25.393418586363367, -4.3238563230628415, 9.606332090203562, 0.9315054065700217, 46.28147088557064, 46.553693107792846, 45.90924866334841, 43.61480421894553, 17.683090541726262, -2.818465926090342, 46.72591533001507, -2.879128689731905, 0.10529635718717323, 1.5340895659339981, 1.956813856697174, 45.26480421890399, 11.289198203807297, 46.63702644112618, 46.56480421890396, 45.07035977445952, 4.04345055916529, 44.98147088557063, 45.74813755223731, 20.198821542632196, -2.4269160759842867, 3.6037364321487946, 45.68702644112618, -5.052496511861257, 47.04813755223729, -3.793204222020833, 40.84813761383674, 45.36480421890413, 6.1875685063100505, 13.287699415545468, 45.798137552237336, 46.39258199668174, -6.201469344866658, 47.998137552237296, 44.8981375522373, 13.12634568369812]
f = [48.61443808689794, -3.214211492697509, 1.4618069219578445, 41.39221624288889, 45.836660309120155, 37.169526159058236, 17.001032883983, 43.30888290124854, 45.92972582430135, 39.23676782977063, 40.671289405737696, 47.653326975786825, -2.215966072526733, 42.25889709221299, 34.60779937989484, 44.23711763683494, 3.9189114678000436, -4.65419661864604, 29.416696803395777, -0.09219966812806213, 41.7520142384852, 44.11571648690499, 6.406521308956398, 46.05888450323387, 46.14787892106885, 38.69120493713146, 11.134772605536703, 1.7792173807445675, 10.372922973311136, 44.45899003217997, 46.947771421824605, 4.634028064534229, 47.875549198009075, 19.256994775690185, 42.76305923074305, 15.515998311075256, 8.918911467816928, 47.92554920972502, 0.674467020372477, 23.361805814057966, 47.2088825313424, 46.28111931312861, 45.49623049895383, 27.31793476416366, 43.269994095407945, 1.335884732660034, -2.60560179306864, 4.180022557561465, 12.34588399538525, 17.234366216886933, -0.5430048514893873, 39.32329427589396, 41.28656249792488, 2.734034084419159, 21.20858572839478, 45.732383163421, 5.411811861671266, 47.836660309120184, 15.989921772871883, 25.973661825201408, 44.01441359251944, 38.66995930325054, 45.34768825100386, 48.331104753564595, 45.06094734189292, -0.30766234711456164, 45.168680904878855, 6.906218900788854, -2.377082120893589, -7.7815846277221405, 40.73399064636388, 42.0225246988825, 47.87564126722249, 1.3510328839122625, 7.331325475199586, 41.34633800969739, 45.55669600396725, 46.09825400009126, 44.90888253134238, 30.84802181445241, 36.82554919800905, 43.04787569917724, 16.01255071245771, 29.001032340629912, -1.6477552242001599, 14.535578134483584, 2.806588439468886, 5.034772936306474, 48.53110475356462, 47.3811047535657, 26.350700725620783, 16.96254834345932, 40.231104806907375, 42.87556338123175, 37.63666729081166, 47.92554952478734, 45.23121137399874, 39.728026789125444, 22.485578116565932, 39.14469396013968]


data_0624_2_IDM_DQN2 = {
    'a': d,
    'b': e,
    'c': f,
}
dqn3_a = [45.08183701757666, 4.01105945037698, 0.0635485338753643, 0.619257457111722, 4.1972299118143646, -1.118749340340841, -1.7661183857589444, 0.23068602908509028, 3.5869103491219665, -0.4956642899415691, -3.987274832737195, 0.07278015281890404, -1.8661081044854697, -0.8501239235758522, -7.160452373188074, 31.75961479696251, -5.431587478187424, 10.334820019435982, -2.3588744854529633, -1.02731229228959, -0.5110306159376066, 3.096396850556216, 3.2821693091492783, 15.78144013797873, 7.739731539664797, 0.26482686417061174, 0.34953149356266877, -0.5349756025237333, -5.46221441253111, -3.831979006946187, 46.254059239798885, -3.6067197775899507, 6.737858897499196, 0.0521712334305402, 3.83387665054528, 5.788081946400455, 5.223214546022923, 5.340848433459348, -0.8988607168394633, -6.330032980555444, -4.6193479412770575, 45.69850368424332, 43.326281462021115, -1.999608560517113, 2.033054037907524, -6.575258468200772, -3.4659519532482097, 41.75961479535445, 4.216747079122573, -0.22938183849566762, 2.254143800019193, 9.816614455600064, -2.644506764706354, 41.215170350909986, -7.224959487644251, 1.6443171365130471, -1.1820613014268773, -6.320517979816722, 45.65961479535443, -0.1569186826466371, 7.027574874217827, 0.4403503900786827, 3.1874907361296003, -8.631632697932291, 5.293850155667863, 41.75961479535444, -2.2453987602327414, 1.602131360464364, 21.598034588289636, -3.9884351294361666, 16.56024620718621, -3.4729443534858913, 16.769750268482063, -6.060324844550793, 10.957750169726665, -4.444422748148093, 0.10388276954176945, 10.795261392493309, -3.594243012656137, 4.8945492374402, 45.870725906465545, 3.5406364818215508, 46.02072590646555, -9.026673860875302, 1.09498944517056, 4.411061979562861, 9.798002678225643, -3.311867138929607, -4.926296078560001, 45.36517035090999, -2.5748041197008504, -7.802907571852793, -0.062063046183707726, 40.942948128687775, 1.551597231186964, -1.5676316223216054, -2.1821946100012095, -7.683311989662387, 3.415702042307423, 2.1667510005551716]

# dqn3_a = [0.09527006326503518, 3.833413268622378, -2.838902346319947, 2.007012649403693, 2.188670973139299, 3.529901931022838, 30.154059239806706, -4.1846544999458875, 45.41517035090999, -4.5229621426490505, 0.8789204252984799, -5.843025465027297, -3.4347723836559103, -3.2817226579510663, -2.9981656077016563, 5.4214396144624235, 46.92072590646555, -2.6566700099968994, -1.683361545972013, -5.806225135328958, 0.3165534879475995, 39.70961479535444, 15.532468429715303, 1.0434672189384617, -3.8275756969908716, 7.146567377011172, 44.20405923979887, -3.9820527798162697, -2.423386233851285, 33.53739257313222, -7.9763888887579855, -1.7877260798113488, 1.5758259803240957, 0.8518791360832676, 3.131840462719209, -3.3514431403327594, 0.3365356996163005, -2.7509460248030813, -2.5340595561114405, 1.5334719562928463, 44.64850368424332, 41.36517035091, 7.3545460407844185, 1.5001041588990773, -7.217346586591794, -3.966773337386541, -2.250851658348699, 0.32897064508979135, -4.957193273217623, 20.53308119859704, -1.8772192083839263, -0.4642651247896694, 39.42628146202111, -7.119555393917846, -5.081925098000838, 7.119909325120034, 2.101226429788369, -4.449753694905504, 34.85961479535444, 6.16890649452813, -0.031147415587344796, -6.60385129771959, 43.931837017576655, 4.683570260767178, 0.8680521362255682, 4.567787966201539, -4.39806690123272, -6.075819296060358, -1.333067606506999, 0.11225322751617917, -2.6620000830495547, 1.1202580903817658, 0.30445044247054476, -1.445639278330546, -4.855719245200198, -1.0012237360150014, 30.95405923980015, -7.588110013803209, 2.563248209076473, 5.414366753584078, -4.444243138644754, -1.101472077903237, -2.1321946100012097, 1.749509110104082, 42.04294812868776, -5.056300171048128, 6.196521643772405, 2.4934389501541556, 27.115187263706378, 2.251642633975205, 3.2813650760020074, -6.799221914912878, 1.5185852261721209, 2.613092690768143, -6.570650786566654, -1.6997817942888904, 2.5283237039344684, 0.06686182519273665, 4.972298559803015, -6.304840791167722]
dqn3_b = [12.024298721218779, 47.06480421890397, 46.07035977445952, -4.906908215172073, 5.82645977419868, -1.0707551238190494, 46.29813755223731, -1.9215790387501723, 45.1981375522373, 45.04813755223732, 45.57035977445952, 45.55924866334843, 47.625915330015076, 44.275915330015074, 43.27035983605989, 41.99258199827504, 45.71480421890397, 43.1259153300151, 46.65369310779286, 45.87591533001509, 38.114804218910514, 45.49813755223731, 44.35924866334843, 3.7718055109150423, 33.60935345599174, 46.120359774459516, 46.9259153300151, 6.609679950655117, 9.512253232135032, 8.009639906375265, -0.19981471681173346, 5.829535777939512, 45.148137552237316, -1.3679867833438246, 40.3417613060678, -5.83765547058353, -2.6980356376758143, 16.144422598009637, -0.139150788152552, -3.2725085031742944, 21.920931322612184, 37.00369310779288, 10.181778209794553, 46.531470885570656, 44.71480459871719, 9.11491267012845, 45.34813755223731, 47.20924866334843, -5.807024955892945, 45.17591533001509, 46.681470885570654, 45.50369310779288, 48.22035977445952, 41.964804218945716, 47.037026441126216, 3.431962625827415, -1.251431296878252, 46.964804218903986, 46.57035977445953, 48.39258199668174, 3.9830719943228168, 46.14813755223731, 46.13702644112621, 45.953693107792866, 45.620359774459686, 0.07304916631527725, 3.626306777107766, 47.01480421890396, 46.29813755223733, 44.125915330015076, 6.266748281233323, 8.876407074161428, 15.011962666628524, 0.4764280833974013, 45.29813755223732, 40.87035977445952, 45.7981375522373, 16.095018665866665, 13.25078818043872, 45.753693107792856, 2.086926360975765, 1.2318609944698, 44.7092486633484, 45.34258199828178, 45.09813755223732, 43.88702650246814, 2.820450271311657, -4.686346794123645, 44.120359774459516, 44.87591533001509, -3.9193396593172105, 46.848137552237304, 8.06108912902616, 0.27719417922413747, -0.6181379171878998, 46.79258199668177, 45.775915330015096, 46.05924866334843, 26.922806269977826, -1.9318374820247524]
dqn3_c = [47.82554919800905, 28.0532266274356, 45.81443808689792, 47.74777142023127, 48.131104753564614, -0.4461540723902875, 42.40350098679761, 48.9588825313424, -0.04390675260134991, -5.146340290428043, 20.66736741776974, 47.21999365231882, 3.627168864454166, 46.858882531600784, 48.11999364245351, 1.979169671231114, -7.496874745758227, 40.05888253134258, 1.690251962273873, 42.01999365426859, 48.0699936424535, 48.15888253290072, 6.0857383770410625, 44.649015478819194, -7.432056032848145, 41.715432483680765, 22.16181185932038, 48.47554919800906, 48.45332697578684, 47.253326975786806, 28.163355912261373, 47.35343125442461, 47.341204484177034, 15.718702472919185, 1.8045491979022472, 47.875549198009054, 47.94777142023128, 47.10888297248991, 48.353326975786835, 43.8934942643681, 48.61443808689792, 48.403326975786825, 47.71443808714962, 32.27921737049562, 48.51443808689795, 46.2311047535648, 47.61445029632884, 44.342215876482975, 49.008882531342394, 25.00103242645835, 48.14777142023128, 20.13436621731632, 8.685572086327094, 49.10888253134239, 48.64221586467571, 32.50454919800899, 40.90098037611249, 24.111811862214203, 48.492215864675714, 47.84777142023124, 48.658882531342385, 7.66769955064966, 23.333843872178836, 48.03110475360524, 47.03666030912581, 47.90888253134237, 43.40942620891249, 10.20633154457345, 48.83666030912016, 48.21999396092478, 14.534034084436405, 47.42554920631607, 47.58666031067848, 41.947774224855195, 47.292215864675725, 43.1200815108057, 46.89777142023128, 47.55389944562731, 12.438406459181959, 48.40332697578684, 27.93002244716095, 11.377731603338933, 47.55332697578683, 48.625549198009224, 9.469526159058239, 47.87554920631607, 18.244398270005593, 48.364438086897934, 46.019993642453485, 47.108882531342374, -1.3159659409140172, 47.93666030912017, 27.824597215116064, 45.67554919800906, 16.817682010252817, 28.01181186219969, 16.150700751103074, 39.46950432672258, -2.1041638571016783, 47.55332697578683]

data_0624_2_IDM_DQN3 = {
    'a': dqn3_a,
    'b': dqn3_b,
    'c': dqn3_c,
}


test = {
    'a': d,
    'b': dqn3_a,
}

df = pd.DataFrame(data_0624_2_IDM_DQN1)
df2 = pd.DataFrame(data_0624_2_IDM_DQN2)
df3 = pd.DataFrame(data_0624_2_IDM_DQN3)
df.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', density = True, label = '直方图')
df.plot(kind = 'kde',  color = 'steelblue',  label = '直方图')
df.plot(kind = 'kde',   label = '直方图')
df.plot.box(title="hua tu")
df2.plot.box(title="hua tu")
df3.plot.box(title="hua tu")
plt.grid(linestyle="--", alpha=0.3)
plt.show()



