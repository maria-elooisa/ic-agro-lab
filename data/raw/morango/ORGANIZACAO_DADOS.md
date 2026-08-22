# Organização dos dados — Morango

Esta estrutura acompanha a lógica usada na pasta do alface, adaptada ao sistema de produção realmente representado pelos dados do morango.

```text
morango/
├── clima/
│   ├── dados-temp.csv
│   └── agrolab_morango_2dias.csv
├── semi_hidro/
│   └── agrolab_morango_dataset.csv
├── solo/
│   └── raw_morango_data.csv
├── graficos/
│   ├── imagens produzidas pela simulação anterior
│   └── imagens geradas pelo EDA_Morango_San_Andreas.ipynb
└── EDA_Morango_San_Andreas.ipynb
```

## Critério de classificação

- `clima/dados-temp.csv`: observações meteorológicas horárias usadas como recorte regional.
- `clima/agrolab_morango_2dias.csv`: agregação e simulação derivada do recorte meteorológico curto.
- `semi_hidro/agrolab_morango_dataset.csv`: série anual sintética utilizada pelo EDA e pelo modelo de ações.
- `solo/raw_morango_data.csv`: geração sintética anterior baseada em temperatura, umidade e nutrientes do solo/substrato.
- `graficos/`: destino único das imagens. Ao executar o EDA novo, os dez gráficos são exportados automaticamente com nomes numerados.

## Por que `semi_hidro` em vez de `nft`?

O morango desta pesquisa está modelado em substrato com fertirrigação. Isso é cultivo semi-hidropônico ou fora do solo, mas não NFT (*Nutrient Film Technique*). Chamar o conjunto de NFT criaria uma inconsistência entre a documentação, as variáveis (`temp_substrato`, `umidade_solo`) e o sistema agronômico estudado.

## Compatibilidade

Os arquivos originais da raiz foram preservados para que os notebooks antigos continuem funcionando. As subpastas contêm cópias organizadas. O notebook `EDA_Morango_San_Andreas.ipynb` já utiliza os novos caminhos.
