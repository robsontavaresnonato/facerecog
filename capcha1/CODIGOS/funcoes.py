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

# pacotes de suporte para ML
from sklearn.externals import joblib

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
		f.write("path, rotulo, caixa_alta_baixa\n")

	for letra in letter_dict:
		for imagem in letter_dict[letra]:
			im = Image.fromarray(imagem)
			im.save("../letras/caracter" + str(i) + ".png")
			with open('../letras.csv', 'a+') as f:
				if letra.istitle():
					f.write("letras/caracter" + str(i) + ".png, " + letra + ", maiusculo\n")
				else:
					if letra.isdigit():
						f.write("letras/caracter" + str(i) + ".png, " + letra + ", numero\n")
					else:
						f.write("letras/caracter" + str(i) + ".png, " + letra + ", minusculo\n")
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
			dic[row['path'].strip()] = {'rotulo':row[' rotulo'].strip(),
										'caixa':row[' caixa_alta_baixa'].strip()}
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

def extract_stats(imgA, imgB):
	mse, iss = compare_images(imgA, imgB)

	mse_centro, iss_centro = compare_images(imgA[10:40,], imgB[10:40,])

	imgA, imgB = imgA[ : , : , 0], imgB[ : , : , 0] # transformada para 2-dimensional para canny e skeleton

	mse_canny, iss_canny = compare_images(feature.canny(imgA, sigma=4), feature.canny(imgB, sigma=4))

	mse_canny_centro, iss_canny_centro = compare_images(feature.canny(imgA[10:40,], sigma=4), feature.canny(imgB[10:40,], sigma=4))

	maskA1, maskB1 = imgA == 255, imgB == 255 # inverter as imagens e deixar em 0s e 1s
	maskA0, maskB0 = imgA == 0, imgB == 0
	imgA[maskA0], imgB[maskB0] = 1, 1
	imgA[maskA1], imgB[maskB1] = 0, 0

	mse_skeleton, iss_skeleton = compare_images(skeletonize(imgA), skeletonize(imgB))

	mse_skeleton_centro, iss_skeleton_centro = compare_images(skeletonize(imgA[10:40,]), skeletonize(imgB[10:40,]))

	return [mse, iss, mse_centro, iss_centro,
			mse_canny, iss_canny, mse_canny_centro, iss_canny_centro,
			mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro]

def save_combinations(permutes, dic, arquivo = "../combinacoes.txt"):
	""" Função que recebe um conjunto de combinações de arquivos e um dicionário com os rótulos.
	O retorno desta função é a criação de um arquivo com as análises de combinações dos arquivos."""
	with open(arquivo, 'w+') as f:
		f.write("resposta,char1,char2,MSE,ISS,MSE_centro,ISS_centro,"+\
		"MSE_canny,ISS_canny,MSE_canny_centro,ISS_canny_centro,"+\
		"MSE_skeleton,ISS_skeleton,MSE_skeleton_centro,ISS_skeleton_centro\n")

	for dupla in permutes:
		imgA = skio.imread("../" + dupla[0])#flatten=True)
		imgB = skio.imread("../" + dupla[1])#, flatten=True)
		if (dic[dupla[0]]['rotulo'] == dic[dupla[1]]['rotulo']):
			resposta = 1
		else:
			resposta = 0
		mse, iss, mse_centro, iss_centro, \
		mse_canny, iss_canny, mse_canny_centro, iss_canny_centro, \
		mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro = extract_stats(imgA, imgB)

		# ISS_skeleton e MSE_canny não aparecem na tabela, é preciso arrumar isso
		with open(arquivo, 'a+') as f:
			f.write(str(resposta) + "," + dupla[0] + "," + dupla[1] + "," + str(mse)\
					+ "," + str(iss) + "," + str(mse_centro) + "," + str(iss_centro)\
					+ "," + str(mse_canny) + "," + str(iss_canny) \
					+ "," + str(mse_canny_centro) + "," + str(iss_canny_centro)\
					+ "," + str(mse_skeleton) + "," + str(iss_skeleton) \
					+ "," + str(mse_skeleton_centro) + "," + str(iss_skeleton_centro) + "\n")

# Fltro nas imagens
def remove_small_blobs(bw_img, min_area=10, **label_kwargs):
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

# Teste do PyTesseract
def run_tesseract(imgs):
	"""Aplica a função image_to_string da biblioteca pytesseract
	em um vetor de funções. O que imprime a String encontrada."""
	for img in imgs:
		img = Image.fromarray( img )
		print( pytesseract.image_to_string( img ) )


# Novas funções
#  função de scoragem:
def super_score(imgA, imgB):

	mse, iss, mse_centro, iss_centro, _, _, _, _, _, _, _, _ = extract_stats(imgA, imgB)

	f =  5.979 -3.201e-05*(mse)  -2.752*(iss) -4.587e-05*(mse_centro)
	prop = exp(f)/(exp(f) + 1)
	score = prop*100

	return score # a funcao retorna um score em formato numerico de 0 a 100.

def super_score2(imgA, imgB):
	# Codigo de Categorização das Variáveis contínuas

	mse, iss, mse_centro, iss_centro, _, _, _, _, _, _, _, _ = extract_stats(imgA, imgB)

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

	f = 7.956e+00 -2.649e-05*(mse)  -2.067e+00*(iss) -4.498e-05*(mse_centro) -4.587e+00*(iss_centro) + CAT_ISS_centro
	prop = exp(f)/(exp(f) + 1)
	score = prop*100

	return score # a funcao retorna um score em formato numerico de 0 a 100.

def super_score3(imgA, imgB):

	mse, iss, mse_centro, iss_centro,\
	mse_canny, iss_canny, mse_canny_centro, iss_canny_centro,\
	mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro = extract_stats(imgA, imgB)

	"""
	(Intercept)          -3.9202
	ISS                   0.8420
	ISS_centro            5.8454
	MSE_canny            -1.7802
	ISS_canny             3.0313
	MSE_canny_centro     17.6900
	ISS_canny_centro      7.0150
	MSE_skeleton        -12.2337
	ISS_skeleton         -4.2933
	MSE_skeleton_centro  -5.2991
	ISS_skeleton_centro  -1.5924
	"""

	f = -3.9202 +0.8420*(iss) +5.8454*(iss_centro) -1.7802*(mse_canny) + \
		3.0313*(iss_canny) +17.6900*(mse_canny_centro) +7.0150*(iss_canny_centro) - \
		12.2337*(mse_skeleton) -4.2933*(iss_skeleton) -5.2991*(mse_skeleton_centro) - \
		1.5924*(iss_skeleton_centro)
	prop = exp(f)/(exp(f) + 1)
	score = prop*100

	return score # a funcao retorna um score em formato numerico de 0 a 100.

# função de busca de letra que dá o melhor matching
def busca_melhor(imgA, v, i, log):

	_, letters_dict = ler_letras("../letras.csv")

	score_ini = 0

	for i in letters_dict:

		imgB = skio.imread("../" + i)

		if v == 1:
			score = super_score(imgA, imgB)
		elif v == 2:
			score = super_score2(imgA, imgB)
		elif v == 3:
			score = super_score3(imgA, imgB)
		else:
			print("v deve estar no intervalo [1, 3].")
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

	clf = joblib.load('classifier' + tipo + '.pkl')

	for imgA in [a, b, c, d, e, f]:
		dic = {}
		for base in letters_dict:

			imgB = skio.imread("../" + base)

			mse, iss, mse_centro, iss_centro,\
			mse_canny, iss_canny, mse_canny_centro, iss_canny_centro,\
			mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro = extract_stats(imgA, imgB)

			if ( clf.predict( [[mse, iss, mse_centro, iss_centro,
			mse_canny, iss_canny, mse_canny_centro, iss_canny_centro,
			mse_skeleton, iss_skeleton, mse_skeleton_centro, iss_skeleton_centro]] ) ):

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
