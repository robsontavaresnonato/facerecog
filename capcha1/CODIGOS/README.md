# Projeto captcha 1

Este projeto foi implementado com Python 3.5, e as bibliotecas utilizadas estão no arquivo [requirements.txt](requirements.txt)

## Procedimento de uso

O arquivo [funcoes.py](funcoes.py) possui diversas funções de suporte implementadas durante o projeto. Estas funções são importadas e usadas nos [notebooks](http://jupyter.org/).

O procedimento que obteve melhores resultados foi:
  1. compilar o notebook [Gerador_base_letras.ipynb](Gerador_base_letras.ipynb)
  2. compilar o notebook [Teste de ML pelas imagens.ipynb](Teste de ML pelas imagens.ipynb)
  3. compilar o notebook [quebrando_captchas.ipynb](quebrando_captchas.ipynb)

Cada notebook possui no seu início algumas variáveis de configuração, como, por exemplo, **v**, **plot** e **save** no notebook *[Gerador_base_letras.ipynb](Gerador_base_letras.ipynb)*. Onde **v** é a versão de filtro à ser aplicada, durante o projeto foram testados vários filtros e o que obteve os melhores resultados foi a versão 2. **plot** igual a *True* fará com que as imagens sejam todas mostradas ao longo do processo do notebook, o que deixa mais devagar o processo e é só necessário caso vá alterar, ou adicionar mais imagens à base de treino, caso contrário mantenha como *False*. E **save** igual *True* habilitará o salvamento dos arquivos das imagens na pasta **../letras** e o arquivo **../letras.csv** que contém o nome do arquivo da imagem e seus labels.

Os outros *notebooks* foram utilizados em etapas anteriores do projeto para testes e exploração de técnicas para resolução do problema.

Observação: O *notebook* [Teste de ML pelas imagens.ipynb](Teste de ML pelas imagens.ipynb) é o mais demorado para processar, pois é onde ocorre o treinamento dos modelos de Aprendizado de Máquina. O melhor modelo é salvo utilizando a função **joblib** (interface da biblioteca **pickle** para objetos da biblioteca **sklearn**), isto cria um arquivo binário (.pkl) que é o objeto (modelo) que é aberto depois no *notebook* [quebrando_captchas.ipynb](quebrando_captchas.ipynb) na chamada da função **modela_captcha()**, implementada no arquivo [funcoes.py](funcoes.py), e realizada a classificação das imagens de validação.

## Resultados obtidos

Na validação alcançamos a precisão de 65% das letras, porém isso representa na prática o acerto de 1 em cada 10 captchas de validação, formados por 6 letras cada.

## Possíveis melhorias futuras

Os algoritmos de Aprendizado de Máquina testados foram **RandomForestClassifier**, **MLPClassifier** e **SVC** todos disponíveis na biblioteca **sklearn**. Foi realizada a otimização dos parâmetros destes algoritmos e também algumas técnicas de *FT* (Feature Transformation) como *PCA* (Principal Component Analysis) e *ICA* (Independent Component Analysis) e de *Image Processing* a técnica *LBP* (Local Binary Pattern), porém nenhuma destas técnicas melhoraram os resultados. Podem ser tentadas novas técnicas de *FT* e foi iniciada a implementação de uma Rede Convolutiva com a biblioteca **tensorflow**, porém não foi finalizada ou testada.
