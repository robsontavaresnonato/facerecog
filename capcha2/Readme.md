

#####################################
# Capcha2
Resolução de Capchas Complexos

# Esta versão de capcha vem com vários aspectos de confusão.
As complexidades são listadas a seguir:

1. Primeiramente temos mais de uma cor.
2. Vários caracteres adcionais em cores mais claras.
3. Imagens rotacionadas

Filtragem
- Alterar para tons de cinza.
- Fazer histograma dos caracteres para pegar o ponto médio de intensidade 
- Eliminar pixels cuja intensidade é menor que o ponto médio menos um valor. Suponto que teremos bits com tons entre 0 a 255, se por exemplo o ponto médio é 154, filtrar os pixels cuja intensidade do nível de cinza seja menor que 140.

Filtro TRESHOLD para eliminar pixels relativos a ruídos e pontos.
