import pickle
import pandas as pd
import collections
import lightgbm as lgb
import numpy as np
import scipy

train = pickle.load(open('dataset/train.pk', 'rb'))
test = pickle.load(open('dataset/test.pk', 'rb'))
dev = pickle.load(open('dataset/dev.pk', 'rb'))

# ans = pickle.load(open('ans.pk', 'rb'))

# for i in range(len(test)):
# 	ans[i]['symptom'] = test[i]['implicit_inform_slots']
# pickle.dump(ans, open('abs.pk', 'wb'))
# zfsym_dic = {}
# i = 2
# with open('dataset/symptom.txt', 'r') as d:
# 	lines = d.readlines()
# 	for line in lines:
# 		zfsym_dic[line.strip()] = i
# 		i += 1
# ks = list(zfsym_dic.keys())
# for k in ks:
# 	zfsym_dic['f' + k] = i
# 	i += 1
# pickle.dump(zfsym_dic, open('zfsym.pkl', 'wb'))
# print(pickle.load(open('zfsym.pkl', 'rb')))
# exit()
disease_dic = {}
disea = []
i = 0
with open('dataset/disease.txt', 'r') as d:
	lines = d.readlines()
	for line in lines:
		disease_dic[line.strip()] = i
		disea.append(line.strip())
		i += 1
sym_dic = {}
syma = []
i = 0
with open('dataset/symptom.txt', 'r') as d:
	lines = d.readlines()
	for line in lines:
		sym_dic[line.strip()] = i
		syma.append(line.strip())
		i += 1
a = [0] * 118
for i in range(len(train)):
	for j in train[i]['explicit_inform_slots'].keys():
		a[sym_dic[j]] += 1
		# if sym_dic[j] == 0:
		# 	print(train[i]['disease_tag'])
	for j in train[i]['implicit_inform_slots'].keys():
		a[sym_dic[j]] += 1
		# if sym_dic[j] == 0:
		# 	print(train[i]['disease_tag'])
xs = np.array(a)
xs = np.max(xs)/(xs + 1)
# isinf = np.isinf(xs)
# xs[isinf] = 0
# xs = np.hstack((xs, xs))

# exfea = []
# imlab = []
# dis_label = []
# for i in range(len(train)):
# 	ex = train[i]['explicit_inform_slots']
# 	im = train[i]['implicit_inform_slots']
# 	exf = [0] * 118
# 	for k in ex.keys():
# 		ind = sym_dic[k]
# 		if ex[k]:
# 			exf[ind] = 1
# 		else:
# 			exf[ind] = -1
# 	iml = [0] * 118
# 	for k in im.keys():
# 		ind = sym_dic[k]
# 		if im[k]:
# 			iml[ind] = 1
# 		else:
# 			iml[ind] = -1
# 	exfea.append(exf)
# 	imlab.append(iml)
# 	dis = train[i]['disease_tag']
# 	disl = [0] * 12
# 	ind = disease_dic[dis]
# 	disl[ind] = 1
# 	dis_label.append(disl)
'''
exfea = []
imlab = []
dis_label = []
for i in range(len(dev)):
	ex = dev[i]['explicit_inform_slots']
	im = dev[i]['implicit_inform_slots']
	exf = [0] * 118
	for k in ex.keys():
		ind = sym_dic[k]
		if ex[k]:
			exf[ind] = 1
		else:
			exf[ind] = -1
	iml = [0] * 118
	for k in im.keys():
		ind = sym_dic[k]
		if im[k]:
			iml[ind] = 1
		else:
			iml[ind] = -1
	exfea.append(exf)
	imlab.append(iml)
	dis = dev[i]['disease_tag']
	disl = [0] * 12
	ind = disease_dic[dis]
	disl[ind] = 1
	dis_label.append(disl)

pickle.dump(exfea, open('devexfea.pkl', 'wb'))
pickle.dump(imlab, open('devimlab.pkl', 'wb'))
pickle.dump(dis_label, open('devdis_label.pkl', 'wb'))
exit()
'''
#---------------

exfea = pickle.load(open('exfea.pkl', 'rb'))
imlab = pickle.load(open('imlab.pkl', 'rb'))
dis_label = pickle.load(open('dis_label.pkl', 'rb'))
devexfea = pickle.load(open('devexfea.pkl', 'rb'))
devimlab = pickle.load(open('devimlab.pkl', 'rb'))
devdis_label = pickle.load(open('devdis_label.pkl', 'rb'))
test = pickle.load(open('dataset/test.pk', 'rb'))

# noim_model = pickle.load(open('xgmodel_noim.pickle.dat', 'rb'))

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance
'''
texfea = []
tdis_label = []
for i in range(len(test)):
	ex = test[i]['explicit_inform_slots']
	im = test[i]['implicit_inform_slots']
	exf = [0] * 118
	for k in ex.keys():
		ind = sym_dic[k]
		if ex[k]:
			exf[ind] = 1
		else:
			exf[ind] = -1
	
	texfea.append(exf)

test_matrix = xgb.DMatrix(np.array(texfea))
pred = noim_model.predict(test_matrix)
for i in range(len(pred)):
	tdis_label.append(round(pred[i]))
	if tdis_label[-1] < 0:
		tdis_label[-1] = 0
	elif tdis_label[-1] > 11:
		tdis_label[-1] = 11
res = []
dic = {}
for i in range(len(tdis_label)):
	dic = {}
	dic['disease'] = disea[tdis_label[i]]
	dic['symptom'] = []
	res.append(dic)
pickle.dump(res, open('ans.pk', 'wb'))
exit()
'''
# from sklearn.preprocessing import Imputer

X = np.array(exfea)

y = np.array(imlab)
yd = np.array(dis_label)

yyd = []
for i in range(len(yd)):
	for j in range(12):
		if yd[i][j] == 1:
			yyd.append(j)
			break
t = np.zeros(X.shape)
t[X == 1] = 1
t[X == -1] = -1
t[y == 1] = 1
t[y == -1] = -1
X = t
# yyda = np.array(yyd)
# for i in range(12):
# 	ind = np.where(yyda == i)[0]
# 	Xt = X[ind, :]
# 	Xsum = np.sum(Xt, axis=0)
# 	ind1 = np.where(Xsum >= ind.shape[0] / 2)[0]
# 	ind2 = np.where(Xsum <= -ind.shape[0] / 2)[0]
# 	if ind1.shape[0] > 0:
# 		for j in range(ind1.shape[0]):
# 			X[ind, ind1[j]] = 1
# 	if ind2.shape[0] > 0:
# 		for j in range(ind2.shape[0]):
# 			X[ind, ind2[j]] = -1

# X = np.hstack((X, y))


devX = np.array(devexfea)

devy = np.array(devimlab)
devyd = np.array(devdis_label)


devyyd = []
for i in range(len(devyd)):
	for j in range(12):
		if devyd[i][j] == 1:
			devyyd.append(j)
			break

fm = 0
def getM(dev):
	global fm
	df = pd.read_csv('C:/Users/dongj/Desktop/xg1p.csv')
	df.set_index('a', inplace=True)
	dftp = pd.read_csv('C:/Users/dongj/Desktop/xg1tp.csv')
	dftp.set_index('a', inplace=True)
	dffp = pd.read_csv('C:/Users/dongj/Desktop/xg1fp.csv')
	dffp.set_index('a', inplace=True)
	dfw = pd.read_csv('C:/Users/dongj/Desktop/xw.csv')
	dfw.set_index('a', inplace=True)
	dft = pd.read_csv('C:/Users/dongj/Desktop/truexgp.csv')
	dft.set_index('a', inplace=True)
	dfft = pd.read_csv('C:/Users/dongj/Desktop/xg1ftp.csv')
	dfft.set_index('a', inplace=True)
	dftt = pd.read_csv('C:/Users/dongj/Desktop/xg1ttp.csv')
	dftt.set_index('a', inplace=True)
	dftft = pd.read_csv('C:/Users/dongj/Desktop/xg1tftp.csv')
	dftft.set_index('a', inplace=True)
	dftff = pd.read_csv('C:/Users/dongj/Desktop/xg1tffp.csv')
	dftff.set_index('a', inplace=True)
	# print(dfw);exit()
	symres = []
	ys = []
	myxs = pickle.load(open('fea_xs.pk', 'rb'))
	px = pickle.load(open('px.pkl', 'rb'))
	csarr = []
	# print(myxs);exit()
	for i in range(len(dev)): #验证集或测试集
		ex = dev[i]['explicit_inform_slots'] #显式
		im = dev[i]['implicit_inform_slots'] #隐式
		symarr = []
		dics = {}
		xwcs = {}
		xw = 0
		cxcs = [1] * 118
		xws = [1] * 22
		sy = np.ones((118,))
		# for _ in range(8):
		for j in ex.keys():
			a = np.array(list(df.loc[:,j].fillna(0)))
			a = a / a.sum()
			b = np.array(list(dftft.loc[:,j].fillna(0)))
			b = b / (b.sum() + 1e-9)
			c = np.array(list(dftff.loc[:,j].fillna(0)))
			c = c / (c.sum() + 1e-9)
			# print(a, b);exit()
			t = np.array(list(df.loc[j,:].fillna(0)))
			if a[a!=0].shape[0] == 0:
				a[a==0]=1/118
			else:
				a[a==0]=min(a[a!=0])#1/118
			if b[b!=0].shape[0] == 0:
				b[b==0]=1/118
			else:
				b[b==0]=min(b[b!=0])#1/118
			if c[c!=0].shape[0] == 0:
				c[c==0]=1/118
			else:
				c[c==0]=min(c[c!=0])#1/118
			t[t==0]=min(t[t!=0])#1/118
			sy *= a * t
			if ex[j]:
				t = np.array(list(dftp.loc[j,:]))
				t[t==0]=min(t[t!=0])#1/118
				sy *= t * b
			else:
				t = np.array(list(dffp.loc[j,:]))
				t[t==0]=min(t[t!=0])#1/118
				sy *= t * c

			for k in range(22):
				xws[k] *= dfw.loc[k, j]
				if dfw.loc[k, j] >= 0.1:

					xw = max(xw, k)
					if k not in xwcs:
						xwcs[k] = 1
					else:
						xwcs[k] += 1
		item = list(xwcs.items())
		cs = min(5, xw - len(list(ex.keys())))#5
		cs = max(4, cs)#3
		# cs = 5

		sy = list(sy)
		
		for j in range(len(sy)):
			if syma[j] in ex.keys():
				sy[j] = 0
		sy = list(enumerate(sy))
		symarr = sorted(sy, key=lambda x: -x[-1])[0:cs]
		
		y = [0] * 118
		ss = []
		k = 0
		ig_cnt = 0
		for j in symarr:
			k += 1
			if syma[j[0]] in im:
				if im[syma[j[0]]]:
					y[j[0]] = 1
				else:
					y[j[0]] = -1
				ig_cnt = 0
			else:
				ig_cnt += 1
			# if ig_cnt == 3:
			# 	break
			ss.append(syma[j[0]])
		fm += k
		symres.append(ss)
		ys.append(y)
	pickle.dump(csarr, open('csarr.pkl', 'wb'))
	return ys, symres

def getMT(dev):
	global fm
	df = pd.read_csv('C:/Users/dongj/Desktop/xg1p.csv', header=0, index_col=0)
	dftp = pd.read_csv('C:/Users/dongj/Desktop/xg1tp.csv', header=0, index_col=0)
	dffp = pd.read_csv('C:/Users/dongj/Desktop/xg1fp.csv', header=0, index_col=0)
	dftft = pd.read_csv('C:/Users/dongj/Desktop/xg1tftp.csv', header=0, index_col=0)
	dftff = pd.read_csv('C:/Users/dongj/Desktop/xg1tffp.csv', header=0, index_col=0)
	symres = []
	ys = []
	myxs = pickle.load(open('fea_xs.pk', 'rb'))
	px = pickle.load(open('px.pkl', 'rb'))
	# print(myxs);exit()
	csarr = pickle.load(open('csarr.pkl', 'rb'))
	for i in range(len(dev)): #验证集或测试集
		ex = dev[i]['explicit_inform_slots'] #显式
		im = dev[i]['implicit_inform_slots'] #隐式
		symarr = []
		dics = {}
		xwcs = {}
		xw = 0
		cxcs = [1] * 118
		xws = [1] * 22
		
		ss = []
		sst = {}
		y = [0] * 118
		ig_cnt = 0
		for ovo in range(1):
			sy = np.ones((118,))
			inv_co = np.ones((118,))
			inv_co1 = np.ones((118,))
			clock = 0
			for j in list(ex.keys()) + list(sst.keys()):

				t = np.array(list(df.loc[j,:].fillna(0)))
				t[t==0]=min(t[t!=0])
				a = np.array(list(df.loc[:,j].fillna(0)))
				a = a / a.sum()
				if a[a!=0].shape[0] == 0:
					a[a==0]=1/118
				else:
					a[a==0]=min(a[a!=0])-1e-9
				if j in sst and sst[j] == -1:
					# t_ind = np.argsort(-t)[0:10]
					# a_ind = np.argsort(-a)[0:10]
					# clock += 1
					# if len(list(sst.keys())) > 2:
					# 	inv_co[t_ind] *= 0.5 
					# 	inv_co1[a_ind] *= 0.5
					# else:
					# 	inv_co[t_ind] *= 2 
					# 	inv_co1[a_ind] *= 2 
					continue
				# elif j in sst and sst[j] != -1:
					# clock = 0
				b = np.array(list(dftft.loc[:,j].fillna(0)))
				b = b / (b.sum() + 1e-9)
				c = np.array(list(dftff.loc[:,j].fillna(0)))
				c = c / (c.sum() + 1e-9)
				
				if b[b!=0].shape[0] == 0:
					b[b==0]=1/118
				else:
					b[b==0]=min(b[b!=0])-1e-9
				if c[c!=0].shape[0] == 0:
					c[c==0]=1/118
				else:
					c[c==0]=min(c[c!=0])-1e-9
				# b[b==0] = 1/118
				# c[c==0] = 1/118
				# sy *= a * t
				if (j in ex and ex[j]) or (j in sst and sst[j]):
					t = np.array(list(dftp.loc[j,:]))
					t[t==0] = min(t[t!=0])-1e-9
					sy *= t * b * inv_co * inv_co1
				else:
					t = np.array(list(dffp.loc[j,:]))
					t[t==0] = min(t[t!=0])-1e-9
					sy *= t * c * inv_co * inv_co1

			cs = 1
			fm += cs

			sy = list(sy)
			
			for j in range(len(sy)):
				if syma[j] in ex.keys() or syma[j] in sst.keys():
					sy[j] = 0
			sy[-1]=0
			sy[-2]=0
			sy = list(enumerate(sy))
			symarr = sorted(sy, key=lambda x: -x[-1])
			
			k = 0
			for j in symarr:
				if k == 1:
					break
				k += 1
				if syma[j[0]] in im:
					if im[syma[j[0]]]:
						y[j[0]] = 1
						sst[syma[j[0]]] = True
					else:
						y[j[0]] = -1
						sst[syma[j[0]]] = False
					ig_cnt = 0
				else:
					ig_cnt += 1
					sst[syma[j[0]]]	= -1
				ss.append(syma[j[0]])
			# if ig_cnt == 5:
			# 	break
		symres.append(ss)
		ys.append(y)
	return ys, symres

#--a--
for epoch in range(2):
	if epoch == 0:
		train_test = 'train'
	else:
		train_test = 'test'

	tX = []
	exfm = 0
	for i in range(len(test)):
		ex = test[i]['explicit_inform_slots']
		t = [0] * 118
		exfm += len(test[i]['implicit_inform_slots'])
		for k in ex.keys():
			if ex[k]:
				t[sym_dic[k]] = 1
			else:
				t[sym_dic[k]] = -1
		tX.append(t)
	tX = np.array(tX)

	if train_test == 'test':
		devX = tX


	# if train_test == 'train':
	# 	devs, symres = getMT(dev)
	# else:
	# 	fm = 0
	# 	devs, symres = getMT(test)
		
	# devs = np.array(devs)
	

	# t = np.zeros(devX.shape)
	# t[devX == 1] = 1
	# t[devX == -1] = -1
	# t[devs == 1] = 1
	# t[devs == -1] = -1
	# devX = t

	# cnt = 0
	# zc = 0
	# fc = 0
	# for i in range(devs.shape[0]):
	# 	f = 0
	# 	z = 0
	# 	for j in range(devs.shape[1]):
	# 		if devs[i][j] == -1:
	# 			f += 1
	# 			fc += 1
	# 		if devs[i][j] == 1:
	# 			z += 1
	# 			zc += 1
	# 	if f == 0 and z == 0:
	# 		cnt += 1
	# print(cnt, zc, fc, zc + fc)
	# if train_test == 'test':
	# 	p = (zc + fc) * 1.0 / fm
	# 	r = (zc + fc) * 1.0 / exfm
	# 	f1 = (1.0/(1.0/p+1.0/r) * 2)
	# 	print(p, r, f1)
	# exit()


	clf = xgb
	if train_test == 'test' or train_test == 'train':
		devyyd = []
		a = [16,20,23,18,27,21,27,20,20,15,17,15]
		for i in range(12):
			devyyd += [i] * a[i]
	a = pickle.load(open('C:/Users/dongj/Desktop/ans.pk', 'rb'))
	res = []
	for i in range(len(a)):
		res.append(disease_dic[a[i]['disease']])
	devyyd = np.array(devyyd)
	res = np.array(res)
	print(res, devyyd)
	print(((res == devyyd) + 0).sum() / res.shape[0]);exit()
	X = scipy.sparse.csr_matrix(X)
	devX = scipy.sparse.csr_matrix(devX)
	train_matrix = clf.DMatrix(X , label=yyd)
	valid_matrix = clf.DMatrix(devX , label=devyyd)

	# params = {'booster': 'gbtree',
	# 			'eval_metric': 'mlogloss',#merror mlogloss
	# 			'gamma': 0.2, #0.2
	# 			'min_child_weight': 0.5,#1.5 0.5
	# 			'max_depth': 10,#5 10
	# 			'lambda':5,#5 
	# 			'subsample': 0.7, #0.7
	# 			'colsample_bytree': 0.1,#0.1
	# 			'eta': 0.05, #0.05

	# 			'seed': 1996,
	# 			'nthread': 36,
	# 			'objective':'multi:softmax',
	# 			'num_class':12,
	# 			}
	params = {'booster': 'gbtree',
				'eval_metric': 'mlogloss',#merror mlogloss
				'gamma': 0.1, #0.2
				'min_child_weight': 0.1,#1.5 0.5
				'max_depth': 12,#5 10
				'lambda':1,#5 
				'subsample': 0.7, #0.7
				'colsample_bytree': 0.1,#0.1
				'eta': 0.05, #0.05

				'seed': 8888,
				'nthread': 36,
				'objective':'multi:softmax',
				'num_class':12,
				}
	if train_test == 'test':
		valid_matrix = clf.DMatrix(devX)
		model = pickle.load(open('xgmodel.pickle.dat', 'rb'))
		val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
		vp = ((val_pred == devyyd) + 0).sum()/len(devyyd)
		print(vp, vp * 0.8 + f1 * 0.2)
	else:
		watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
		model = clf.train(params, train_matrix, num_boost_round=5000, evals=watchlist, verbose_eval=50, early_stopping_rounds=200)
		val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
		vp = ((val_pred == devyyd) + 0).sum()/len(devyyd)
		print(vp);exit()
		pickle.dump(model, open('xgmodel.pickle.dat', 'wb'))
	if train_test == 'test':
		ans = []
		exit()
		for i in range(len(test)):
			dic = {}
			dic['disease'] = disea[int(val_pred[i])]
			dic['symptom'] = symres[i]
			ans.append(dic)
		pickle.dump(ans, open('ansmy.pk', 'wb'));exit()


import lightgbm as lgb
from sklearn import metrics 
from sklearn.model_selection import GridSearchCV

params = {  
	'n_estimators':100,
    'boosting_type': 'gbdt',  
    'objective': 'multiclass',  
    'num_class': 12,
    'metric': 'multi_error',  
    'num_leaves': 10,  
    'min_data_in_leaf': 5, 
    'max_depth': 4,
    # 'maxbin': 2,
    # "min_sum_hessian_in_leaf": 2, 
    'learning_rate': 0.3, #0.3 
    'feature_fraction': 0.1, #0.1 
    'bagging_fraction': 0.9,  #0.8
    'bagging_freq': 5,  
    # 'lambda_l1': 0.05,
    'lambda_l2': 4,
    'min_gain_to_split': 0.6,  #0.6
    'verbose': -1, 
}


if train_test == 'test':
	devX = scipy.sparse.csr_matrix(devX)
	clf = pickle.load(open('lgbmodel.pickle.dat', 'rb'))
	val_pred = clf.predict(devX, num_iteration=clf.best_iteration)
	val_pred = [list(x).index(max(x)) for x in val_pred]
	print(((np.array(val_pred) == devyyd) + 0).sum()/len(devyyd));exit()
	ans = []
	for i in range(len(test)):
		dic = {}
		dic['disease'] = disea[int(val_pred[i])]
		dic['symptom'] = symres[i]
		ans.append(dic)
	pickle.dump(ans, open('ansmy.pk', 'wb'));exit()



X = scipy.sparse.csr_matrix(X)
devX = scipy.sparse.csr_matrix(devX)
trn_data = lgb.Dataset(X, yyd)
val_data = lgb.Dataset(devX, devyyd)
clf = lgb.train(params, 
	trn_data, 
	num_boost_round = 1000,
	valid_sets = [trn_data,val_data], 
	verbose_eval = 50, 
	early_stopping_rounds = 200)
y_prob = clf.predict(devX, num_iteration=clf.best_iteration)
print(clf.predict(devX, num_iteration=clf.best_iteration, pred_leaf=True)[0].shape)
y_pred = [list(x).index(max(x)) for x in y_prob]

print("AUC score: {:<8.5f}".format(metrics.accuracy_score(y_pred, devyyd)))
pickle.dump(clf, open('lgbmodel.pickle.dat', 'wb'));


