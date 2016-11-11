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
from skimage.util import dtype_limits
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.filters import rank
from skimage.measure import compare_ssim, compare_mse

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
    if plot == False:
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

def save_combinations(permutes, dic, arquivo = "../combinacoes.txt"):
    """ Função que recebe um conjunto de combinações de arquivos e um dicionário com os rótulos.
    O retorno desta função é a criação de um arquivo com as análises de combinações dos arquivos."""
    with open(arquivo, 'w+') as f:
        f.write("resposta,char1,char2,MSE,ISS,MSE_centro,ISS_centro\n")

    for dupla in permutes:
        imgA = skio.imread("../" + dupla[0])
        imgB = skio.imread("../" + dupla[1])
        if (dic[dupla[0]]['rotulo'] == dic[dupla[1]]['rotulo']):
            resposta = 1
        else:
            resposta = 0
        m, s = compare_images(imgA, imgB)

        mc, cs = compare_images(imgA[10:40,], imgB[10:40,])

        with open(arquivo, 'a+') as f:
            f.write(str(resposta) + "," + dupla[0] + "," + dupla[1] + "," + str(m) + \
                    "," + str(s) + "," + str(mc) + "," + str(cs) + "\n")

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
def super_score(MSE, ISS, MSE_centro, ISS_centro):
	# aqui vai o código
	f =  5.979 -3.201e-05*(MSE)  -2.752*(ISS) -4.587e-05*(MSE_centro)
	prop = exp(f)/(exp(f) + 1)
	score = prop*100

	return score # a funcao retorna um score em formato numerico de 0 a 100.

def super_score2(MSE, ISS, MSE_centro, ISS_centro):
	# Codigo de Categorização das Variáveis contínuas

    if (ISS_centro > 0 and ISS_centro <= 0.20468457663):
        CAT_ISS_centro = 0
    elif (ISS_centro > 0.20468457663 and ISS_centro <= 0.34859473007):
        CAT_ISS_centro = -8.073e-01
    elif (ISS_centro > 0.34859473007 and ISS_centro <= 0.384041534944):
        CAT_ISS_centro = -2.944e-01
    elif (ISS_centro > 0.384041534944 and ISS_centro <= 0.444163127529):
        CAT_ISS_centro = 8.699e-02
    elif (ISS_centro > 0.444163127529):
        CAT_ISS_centro = 1.506e+00
    else:
        CAT_ISS_centro = 0
	     
    if (MSE_centro > 0 and MSE_centro <= 49419):
        CAT_MSE_centro = 0
    elif (MSE_centro > 49419 and MSE_centro <= 59079.8571429):
        CAT_MSE_centro = -8.073e-01
    elif (MSE_centro > 59079.8571429 and MSE_centro <= 65953.9285714):
        CAT_MSE_centro = -2.944e-01
    elif (MSE_centro > 65953.9285714 and MSE_centro <= 76172.1428571):
        CAT_MSE_centro = 8.699e-02
    elif (MSE_centro > 76172.1428571 and MSE_centro <= 85275.6428571):
        CAT_MSE_centro = 1.506e+00
    elif (MSE_centro > 85275.6428571):
        CAT_MSE_centro = 0
    else:
        CAT_MSE_centro = 0
             
    if (ISS > 0 and ISS <= 0.395029459251):
        CAT_ISS = 0
    elif (ISS > 0.395029459251 and ISS <= 0.518022856003):
        CAT_ISS = -8.073e-01
    elif (ISS > 0.518022856003 and ISS <= 0.549536501022):
        CAT_ISS = -2.944e-01	
    elif (ISS > 0.549536501022):
        CAT_ISS = 8.699e-02
    else:
        CAT_ISS = 0
             
    if (MSE > 0 and MSE <= 40129.7142857):
        CAT_MSE = 0
    elif (MSE > 40129.7142857 and MSE <= 44774.3571429):
        CAT_MSE = -8.073e-01
    elif (MSE > 44774.3571429 and MSE <= 47969.8714286):
        CAT_MSE = -2.944e-01
    elif (MSE > 47969.8714286 and MSE <= 54658.1571429):
        CAT_MSE = 8.699e-02
    elif (MSE > 54658.1571429 and MSE <= 60937.7142857):
        CAT_MSE = 1.506e+00
    elif (MSE > 60937.7142857):
        CAT_MSE = 0
    else:
        CAT_MSE = 0

    f = 7.956e+00 -2.649e-05*(MSE)  -2.067e+00*(ISS) -4.498e-05*(MSE_centro) -4.587e+00*(ISS_centro) + CAT_MSE + CAT_ISS + CAT_MSE_centro + CAT_ISS_centro
    prop = exp(f)/(exp(f) + 1)
    score = prop*100

    return score # a funcao retorna um score em formato numerico de 0 a 100.

# função de busca de letra que dá o melhor matching
def busca_melhor(letra):

	_, letters_dict = ler_letras("../letras.csv")
	
	score_ini = 0

	for i in letters_dict:
		
		img = skio.imread("../" + i)
		
		m, s = compare_images(letra, img)
		mc, cs = compare_images(letra[10:40,], img[10:40,])
	 	 
		score = super_score(m, s, mc, cs)
		if score > score_ini: # then letra_oficial=dicionario[i]
			rotulo_letra_maior_score = str(letters_dict[i]['rotulo'])
			score_ini = score

	return rotulo_letra_maior_score

# função de busca de letra que dá o melhor matching
def busca_melhor2(letra):

	_, letters_dict = ler_letras("../letras.csv")
	
	score_ini = 0

	for i in letters_dict:
		
		img = skio.imread("../" + i)
		
		m, s = compare_images(letra, img)
		mc, cs = compare_images(letra[10:40,], img[10:40,])
	 	 
		score = super_score2(m, s, mc, cs)
		if score > score_ini: # then letra_oficial=dicionario[i]
			rotulo_letra_maior_score = str(letters_dict[i]['rotulo'])
			score_ini = score

	return rotulo_letra_maior_score


def quebra_captcha(captcha):

	a = crop_char(captcha, 0)
	b = crop_char(captcha, 1)
	c = crop_char(captcha, 2)
	d = crop_char(captcha, 3)
	e = crop_char(captcha, 4)
	f = crop_char(captcha, 5)
	
	resposta = ""

	for letra in [a, b, c, d, e, f]:
		melhor = busca_melhor(letra)
		resposta = resposta + melhor
	return resposta

def quebra_captcha2(captcha):

	a = crop_char(captcha, 0)
	b = crop_char(captcha, 1)
	c = crop_char(captcha, 2)
	d = crop_char(captcha, 3)
	e = crop_char(captcha, 4)
	f = crop_char(captcha, 5)
	
	resposta = ""

	for letra in [a, b, c, d, e, f]:
		melhor = busca_melhor2(letra)
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
	
	for letra in [a, b, c, d, e, f]:
		dic = {}
		for base in letters_dict:
			
			img = skio.imread("../" + base)

			m, s = compare_images(letra, img)
			cm, cs = compare_images(letra[10:40,], img[10:40,])

			if ( clf.predict( [[m, s, cm, cs]] ) ):
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
