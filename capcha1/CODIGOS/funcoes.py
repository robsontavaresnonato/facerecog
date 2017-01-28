# funções.py
"""
Contém as principais funcões utilizadas para o projeto
capcha1.
"""

# handling files support packages
from glob import glob

# logic support packages
import numpy as np
import pytesseract
import itertools
import csv
import pandas as pd
from math import exp
import operator

# plot support packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# image trasformation packages
from PIL import Image
import skimage.io as skio
from skimage import feature
from skimage.util import dtype_limits
from skimage.morphology import label, skeletonize
from skimage.measure import regionprops
from skimage.filters import rank
from skimage.measure import compare_ssim, compare_mse
from sklearn.preprocessing import binarize
from skimage.morphology import dilation, erosion
from skimage.morphology import disk

# pacotes de suporte para ML
from sklearn.externals import joblib
from sklearn import decomposition

def apply_filter(img, v = 1):
	if v == 1:
		return remove_small_blobs(img[ : , : , 0], background=255)
	elif v == 2:
		selem = disk(1.4)
		dilatado = dilation(remove_small_blobs(img[ : , : , 0], background=255, min_area=10), selem)
		unblobbed2 = remove_small_blobs(erosion(dilatado, selem), background=255, min_area=15)
		return rank.mean(unblobbed2, selem=selem)
	elif v == 3:
		unblobbed = remove_small_blobs(img[ : , : , 0], background=255)
		selem = disk(1.4)
		return remove_small_blobs(dilation(unblobbed, selem), background=255, min_area=10)
	else:
		pass

# Funções de comparação entre imagens
def mse(imageA, imageB):
	"""Função de compara entre duas imagens e retorna o erro médio quadrado
	entre as duas imagens.
	Nota: as imagens devem ter as mesmas dimensões"""
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def compare_images(imageA, imageB, title = "Titulo", channel=True, plot=False):
	"""Função para comparar duas imagens e retornar um plot
	ou as medidas mse e ssim das imagens, o que depende de
	plot=True ou plot=False respectivamente."""
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = compare_ssim(imageA, imageB, multichannel=channel)
	if not plot:
		return [m, s]
	else:
		# setup the figure
		fig = plt.figure(title)
		plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

		# show first image
		ax = fig.add_subplot(1, 2, 1)
		plt.imshow(imageA, cmap = plt.cm.gray)
		plt.axis("off")

		# show the second image
		ax = fig.add_subplot(1, 2, 2)
		plt.imshow(imageB, cmap = plt.cm.gray)
		plt.axis("off")

		# show the images
		plt.show()

# Funções de suporte
def load_image( infilename ) :
	img = Image.open( infilename )
	img.load()
	data = np.asarray( img, dtype="int32" )
	return data

def save_image( npdata, outfilename ) :
	img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
	img.save( outfilename )

def plot_captchas(imgs, grid=(5, 4)):
	gs = gridspec.GridSpec(*grid)
	for idx, gspec in enumerate(gs):
		ax = plt.subplot(gspec)
		plt.imshow(imgs[idx])

def save_images(letter_dict):
	i=0

	with open('../letras.csv', 'w+') as f:
		f.write("path,rotulo,caixa_alta_baixa\n")

	for letra in letter_dict:
		for imagem in letter_dict[letra]:
			im = Image.fromarray(imagem)
			im.save("../letras/caracter" + str(i) + ".png")
			with open('../letras.csv', 'a+') as f:
				if letra.istitle():
					f.write("../letras/caracter" + str(i) + ".png," + letra + ",maiusculo\n")
				else:
					if letra.isdigit():
						f.write("../letras/caracter" + str(i) + ".png," + letra + ",numero\n")
					else:
						f.write("../letras/caracter" + str(i) + ".png," + letra + ",minusculo\n")
			i = i + 1

def crop_char(img, n_char, x1 = 0, x2 = 50):
	sw = {
		0: {'x1': x1, 'x2': x2, 'y1': 5, 'y2': 40},
		1: {'x1': x1, 'x2': x2, 'y1': 35, 'y2': 70},
		2: {'x1': x1, 'x2': x2, 'y1': 65, 'y2': 100},
		3: {'x1': x1, 'x2': x2, 'y1': 95, 'y2': 130},
		4: {'x1': x1, 'x2': x2, 'y1': 123, 'y2': 158},
		5: {'x1': x1, 'x2': x2, 'y1': 145, 'y2': 180},
	}
	return img[sw[n_char]['x1']:sw[n_char]['x2'], sw[n_char]['y1']:sw[n_char]['y2']]

def feed_char_dict(char_dict, letter_array, imgs):
	for letter, img in zip(letter_array, imgs):

		if letter in char_dict:
			char_dict[letter].append(img)
		else:
			char_dict[letter] = [img]

def ler_letras(file):
	""" Função para ler o arquivo letras.csv.
	Retorno duplo de uma lista de arquivos e um dicionário dos arquivos rótulo e caixa"""
	dic = {}
	with open(file) as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',')
		for row in reader:
			dic[row['path'].strip()] = {'rotulo':row['rotulo'],
										'caixa':row['caixa_alta_baixa']}
		return [list(dic.keys()), dic]

def checar_combinacoes(permuta, dic):
	""" Função que recebe a permutação de arquivos e dicionário com rótulos
	e retorna, em uma lista, as quantidades de permutações com rótulos iguais e
	a quantidade de permutações com rótulos diferentes."""
	mesmo = 0.
	diferentes = 0.
	for i in permuta:
		if (dic[i[0]]['rotulo'] == dic[i[1]]['rotulo']):
			mesmo = mesmo + 1
		else:
			diferentes = diferentes + 1
	return [mesmo, diferentes]

# Fltro nas imagens
def remove_small_blobs(bw_img, min_area=35, **label_kwargs):
	""" Remove small blobs in the bw img. """
	labels = label(bw_img, **label_kwargs)

	# pick the background and foreground colors
	bg = label_kwargs.get('background', 0)
	fg = dtype_limits(bw_img, clip_negative=True)[1] - bg

	# create an empty image
	new_bw = np.ones_like(bw_img) * bg

	# check the area of each region
	for roi in regionprops(labels):
		if roi.area >= min_area:
			new_bw[labels == roi.label] = fg

	return new_bw

def entropy(signal):
		'''
		function returns entropy of a signal
		signal must be a 1-D numpy array
		'''
		lensig=len(signal)#.size
		symset=list(set(signal))
		numsym=len(symset)
		propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
		ent=np.sum([p*np.log2(1.0/p) for p in propab])
		return ent

def extract_stats(imgA, imgB, sigma = 1):
	imgA = remove_small_blobs(imgA, background = 255)
	imgB = remove_small_blobs(imgB, background = 255)

	mse, iss = compare_images(imgA, imgB)
	mse_centro, iss_centro = compare_images(imgA[10:40,], imgB[10:40,])

	if (len(imgA.shape) == 3):
		imgA, imgB = imgA[ : , : , 0], imgB[ : , : , 0] # transformada para 2-dimensional para canny e skeleton

	mse_canny, iss_canny = compare_images(feature.canny(imgA, sigma=sigma), feature.canny(imgB, sigma=sigma))
	mse_canny_centro, iss_canny_centro = compare_images(feature.canny(imgA[10:40,], sigma=4), feature.canny(imgB[10:40,], sigma=4))

	maskA1, maskB1 = imgA == 255, imgB == 255 # inverter as imagens e deixar em 0s e 1s
	maskA0, maskB0 = imgA == 0, imgB == 0
	imgA[maskA0], imgB[maskB0] = 1, 1
	imgA[maskA1], imgB[maskB1] = 0, 0
	mse_skeleton, iss_skeleton = compare_images(skeletonize(imgA), skeletonize(imgB))
	mse_skeleton_centro, iss_skeleton_centro = compare_images(skeletonize(imgA[10:40,]), skeletonize(imgB[10:40,]))

	# mean
	imgA_mean, imgB_mean = [imgA.mean(), imgB.mean()]
	# variancia
	imgA_var, imgB_var = [imgA.var(), imgB.var()]
	# Contraste
	"imgA_contraste, imgB_contraste ="
	# Segundo Momento Angular
	"imgA_angular_momentum, imgB_angular_momentum ="
	# Entropia
	imgA_entropy, imgB_entropy = [entropy([item for sublist in imgA.tolist() for item in sublist]),
								entropy([item for sublist in imgB.tolist() for item in sublist])]

	return [mse, iss, mse_centro, iss_centro,
			mse_canny, iss_canny, mse_canny_centro, iss_canny_centro,
			mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro,
			imgA_mean, imgB_mean, imgA_var, imgB_var,
			#imgA_contraste, imgB_contraste,
			#imgA_angular_momentum, imgB_angular_momentum,
			imgA_entropy, imgB_entropy]


def save_combinations(permutes, dic, arquivo = "../combinacoes.txt"):
	""" Função que recebe um conjunto de combinações de arquivos e um dicionário com os rótulos.
	O retorno desta função é a criação de um arquivo com as análises de combinações dos arquivos."""
	with open(arquivo, 'w+') as f:
		f.write("resposta,char1,char2,mse,iss,mse_centro,iss_centro,"+\
		"mse_canny,iss_canny,mse_canny_centro,iss_canny_centro,"+\
		"mse_skeleton,iss_skeleton,mse_skeleton_centro,iss_skeleton_centro,"+\
		"imgA_mean,imgB_mean,imgA_var,imgB_var,"+\
		#"imgA_contraste,imgB_contraste,imgA_angular_momentum,imgB_angular_momentum,"+\
		"imgA_entropy,imgB_entropy\n")

	for dupla in permutes:
		imgA = skio.imread(dupla[0])
		imgB = skio.imread(dupla[1])
		if (dic[dupla[0]]['rotulo'] == dic[dupla[1]]['rotulo']):
			resposta = 1
		else:
			resposta = 0

		mse, iss, mse_centro, iss_centro, \
		mse_canny, iss_canny, mse_canny_centro, iss_canny_centro, \
		mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro, \
		imgA_mean, imgB_mean, imgA_var, imgB_var, \
		imgA_entropy, imgB_entropy = extract_stats(imgA, imgB)
		#imgA_contraste, imgB_contraste, imgA_angular_momentum, imgB_angular_momentum, \

		# ISS_skeleton e MSE_canny não aparecem na tabela, é preciso arrumar isso
		with open(arquivo, 'a+') as f:
			f.write(str(resposta) + "," + dupla[0] + "," + dupla[1] + "," + str(mse)\
					+ "," + str(iss) + "," + str(mse_centro) + "," + str(iss_centro)\
					+ "," + str(mse_canny) + "," + str(iss_canny) \
					+ "," + str(mse_canny_centro) + "," + str(iss_canny_centro)\
					+ "," + str(mse_skeleton) + "," + str(iss_skeleton) \
					+ "," + str(mse_skeleton_centro) + "," + str(iss_skeleton_centro) \
					+ "," + str(imgA_mean) + "," + str(imgB_mean) \
					+ "," + str(imgA_var) + "," + str(imgB_var) \
					#+ "," + str(imgA_contraste) + "," + str(imgB_contraste) \
					#+ "," + str(imgA_angular_momentum) + "," + str(imgB_angular_momentum) \
					+ "," + str(imgA_entropy) + "," +  str(imgB_entropy) + "\n")

# Teste do PyTesseract
def run_tesseract(imgs):
	"""Aplica a função image_to_string da biblioteca pytesseract
	em um vetor de funções. O que imprime a String encontrada."""
	for img in imgs:
		img = Image.fromarray( img )
		print( pytesseract.image_to_string( img ) )


# Novas funções
#  função de scoragem:
def super_score(imgA, imgB, v):
	mse, iss, mse_centro, iss_centro, \
	mse_canny, iss_canny, mse_canny_centro, iss_canny_centro, \
	mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro, \
	imgA_mean, imgB_mean, imgA_var, imgB_var, \
	imgA_entropy, imgB_entropy = extract_stats(imgA, imgB)
	#imgA_contraste, imgB_contraste, imgA_angular_momentum, imgB_angular_momentum, \

	if (v == 1):
		f =  5.979 -3.201e-05*(mse)  -2.752*(iss) -4.587e-05*(mse_centro)

	elif(v == 2):
		if (iss_centro > 0 and iss_centro <= 0.20468457663):
			CAT_ISS_centro = 0
		elif (iss_centro > 0.20468457663 and iss_centro <= 0.34859473007):
			CAT_ISS_centro = -8.073e-01
		elif (iss_centro > 0.34859473007 and iss_centro <= 0.384041534944):
			CAT_ISS_centro = -2.944e-01
		elif (iss_centro > 0.384041534944 and iss_centro <= 0.444163127529):
			CAT_ISS_centro = 8.699e-02
		elif (iss_centro > 0.444163127529):
			CAT_ISS_centro = 1.506e+00
		else:
			CAT_ISS_centro = 0
		f = (7.956e+00 -2.649e-05*(mse)  -2.067e+00*(iss) -4.498e-05*(mse_centro)
			-4.587e+00*(iss_centro) + CAT_ISS_centro)

	elif(v == 3):
		f = (2.98 -0.64*iss_centro - 6.92*mse_canny -2.57*iss_canny
			+ 9.99*mse_canny_centro +3.01*iss_canny_centro
			-0.78*iss_skeleton +2.86*mse_skeleton_centro
			-0.83*iss_skeleton_centro)

	elif(v == 4):
		f = (2.960e+00 + 1.203e-04 * (mse) -1.915e-04 * (mse_centro)
					-2.153e+00 * (iss) -4.615e+00*(iss_centro)
					+ 3.612e+00 * (mse_canny) + 2.380e+00 * (iss_canny)
					+ 2.618e+00 * (iss_canny_centro) -4.454e+01 * (mse_skeleton)
					-1.958e+00*(iss_skeleton) + 5.035e+01*(mse_skeleton_centro) + 4.036e+00*(iss_skeleton_centro ))

	elif(v == 5):
		f = (3.099e+00 + 4.057e-05 * (mse) -1.209e-04 * (mse_centro) -2.237e+00 * (iss)
					-1.879e+00*(iss_centro) -1.695e+00 * (iss_canny) + 1.167e+01 * (mse_canny_centro) + 2.626e+00* (iss_canny_centro)
					-2.132e+01 * (mse_skeleton) + 3.390e+00*(iss_skeleton) + 2.387e+01*(mse_skeleton_centro) -1.636e+00 *(iss_skeleton_centro ))

	elif(v == 6):
		var2 = (imgA_mean - imgB_mean)**2
		f = (-3.001e+00 + 1.046e-03*mse - 1.046e-03*mse_centro + 1.216e+01*iss - 1.476e+01*iss_centro
		+ 2.173e+01*mse_canny + 1.550e+01*iss_canny + 3.034e+00*mse_canny_centro - 1.618e+02*mse_skeleton
		- 2.826e+01*iss_skeleton + 1.093e+02*mse_skeleton_centro + 2.115e+01*iss_skeleton_centro
		+ 1.003e+02*imgA_mean + 5.918e+01*imgB_mean - 9.723e+01*imgA_var - 4.455e+01*imgB_var
		- 4.463e+01*var2)
	prop = exp(f)/(exp(f) + 1)
	score = prop*100
	return score # a funcao retorna um score em formato numerico de 0 a 100.


# função de busca de letra que dá o melhor matching
def busca_melhor(imgA, v, i, log):
	_, letters_dict = ler_letras("../letras.csv")
	score_ini = 0
	for i in letters_dict:
		imgB = skio.imread(i)
		if v in [1,2,3,4,5,6]:
			score = super_score(imgA, imgB, v)
		else:
			print("v deve estar no intervalo [1, 6].")
			break
		if score > score_ini: # then letra_oficial=dicionario[i]
			rotulo_letra_maior_score = str(letters_dict[i]['rotulo'])
			score_ini = score
			print("letra {0} - novo melhor é : {1}.".format(i, score_ini))

	return rotulo_letra_maior_score


def quebra_captcha(captcha, v, log = False):
	a = crop_char(captcha, 0)
	b = crop_char(captcha, 1)
	c = crop_char(captcha, 2)
	d = crop_char(captcha, 3)
	e = crop_char(captcha, 4)
	f = crop_char(captcha, 5)

	resposta = ""
	for i, letra in enumerate([a, b, c, d, e, f]):
		melhor = busca_melhor(letra, v, i, log)
		resposta = resposta + melhor

	return resposta


def modela_captcha(captcha, tipo = ""):
	_, letters_dict = ler_letras("../letras.csv")
	a = crop_char(captcha, 0)
	b = crop_char(captcha, 1)
	c = crop_char(captcha, 2)
	d = crop_char(captcha, 3)
	e = crop_char(captcha, 4)
	f = crop_char(captcha, 5)

	resposta = ""
	clf = joblib.load('classifier_' + tipo + '.pkl')
	if ( tipo != "image" ):
		for imgA in [a, b, c, d, e, f]:
			dic = {}
			for base in letters_dict:

				imgB = skio.imread(base)

				mse, iss, mse_centro, iss_centro, \
				mse_canny, iss_canny, mse_canny_centro, iss_canny_centro, \
				mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro, \
				imgA_mean, imgB_mean, imgA_var, imgB_var, \
				imgA_entropy, imgB_entropy = extract_stats(imgA, imgB)
				#imgA_contraste, imgB_contraste, imgA_angular_momentum, imgB_angular_momentum, \

				if ( clf.predict([ [mse, iss, mse_centro, iss_centro, \
				mse_canny, iss_canny, mse_canny_centro, iss_canny_centro, \
				mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro, \
				imgA_mean, imgB_mean, imgA_var, imgB_var, \
				#imgA_contraste, imgB_contraste, \
				#imgA_angular_momentum, imgB_angular_momentum, \
				imgA_entropy, imgB_entropy] ]) ):

					if (letters_dict[base]['rotulo'] in dic):
						dic[ letters_dict[base]['rotulo'] ] += 1
					else:
						dic[ letters_dict[base]['rotulo'] ] = 1

			if (dic):
				melhor = max(dic.items(), key=operator.itemgetter(1))[0]
				resposta = resposta + melhor
			else:
				resposta = resposta + " "

		return resposta
	else :
		for imgA in [a, b, c, d, e, f]:
			vetor_imagem = [item for sublist in imgA.tolist() for item in sublist]
			resposta = resposta + str(clf.predict(vetor_imagem)[0])
		return resposta

def tsrct_captcha(captcha):
	_, letters_dict = ler_letras("../letras.csv")
	a = crop_char(captcha, 0)
	b = crop_char(captcha, 1)
	c = crop_char(captcha, 2)
	d = crop_char(captcha, 3)
	e = crop_char(captcha, 4)
	f = crop_char(captcha, 5)

	resposta = ""
	for letra in [a, b, c, d, e, f]:

		img = Image.fromarray( letra )
		tesser =  pytesseract.image_to_string( img )

		if (tesser == ""):
			resposta = resposta + " "
		else:
			resposta = resposta + tesser

	return resposta
