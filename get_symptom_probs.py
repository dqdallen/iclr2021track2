import numpy as np
import pandas as pd
import pickle

def get_disease(path=''):
	disease_dic = {}
	disea = []
	i = 0
	with open(path + '/disease.txt', 'r') as d:
		lines = d.readlines()
		for line in lines:
			disease_dic[line.strip()] = i
			disea.append(line.strip())
			i += 1
	return disease_dic, disea


def get_symptom(path=''):
	sym_dic = {}
	syma = []
	i = 0
	with open(path + '/symptom.txt', 'r') as d:
		lines = d.readlines()
		for line in lines:
			sym_dic[line.strip()] = i
			syma.append(line.strip())
			i += 1
	return sym_dic, syma


def get_related_probs(sour_path='', des_path=''):
	train = pickle.load(open(sour_path + '/train.pk', 'rb'))
	disease_dic, disea = get_disease(sour_path)
	sym_dic, syma = get_symptom(sour_path)
	for types in ['tftp', 'tffp', 'fp', 'tp', 'p']:
		cnt = 0
		dic = {}
		for ind in range(len(syma)):
			dic[syma[ind]] = [0] * 118
		df = pd.DataFrame(dic)
		df.index = syma
		for j in list(sym_dic.keys()):
			for k in list(sym_dic.keys()):
				cnt = 0
				if j == k:
					continue
				for i in range(len(train)):
					ex = list(train[i]['explicit_inform_slots'].keys())
					im = list(train[i]['implicit_inform_slots'].keys())
					e = train[i]['explicit_inform_slots']
					m = train[i]['implicit_inform_slots']
					if types == 'tffp':
						if (j in ex and k in im and not m[k]) or (j in im and k in ex  and not e[k]) or \
						 (j in im and k in im and  not m[k]) or (j in ex and k in ex  and not e[k]):
							cnt += 1
					elif types == 'tftp':
						if (j in ex and k in im and m[k]) or (j in im and k in ex  and e[k]) or \
						 (j in im and k in im and  m[k]) or (j in ex and k in ex  and e[k]):
							cnt += 1
					elif types == 'fp':
						if (j in ex and k in im and not e[j]) or (j in im and k in ex  and not m[j]) or \
						 (j in im and k in im and  not m[j]) or (j in ex and k in ex  and not e[j]):
							cnt += 1
					elif types == 'tp':
						if (j in ex and k in im and e[j]) or (j in im and k in ex  and m[j]) or \
						 (j in im and k in im and  m[j]) or (j in ex and k in ex  and e[j]):
							cnt += 1
					elif types == 'p':
						if (j in ex and k in im) or (j in im and k in ex) or \
						 (j in im and k in im) or (j in ex and k in ex):
							cnt += 1
				df.loc[j, k] = cnt
		df = df.div(df.sum(axis=1), axis=0)
		df.to_csv(des_path + '/xg'+types+'.csv')
# get_related_probs('comp', 'comp')