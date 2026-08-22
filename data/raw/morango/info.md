# Organização dos dados — Morango

```text
morango/
├── clima/
│   ├── dados-temp.csv
│   ├── agrolab_morango_2dias.csv
│   ├── clima_processado.csv
│   └── relatorio_clima.txt
├── nft/
│   ├── dataset_nft.csv
│   └── relatorio_dataset_nft.txt
├── semi_hidro/
│   └── agrolab_morango_dataset.csv
├── solo/
│   ├── raw_morango_data.csv
│   ├── dataset_solo.csv
│   └── relatorio_dataset_solo.txt
├── graficos/
├── pipeline_geracao_morango.ipynb
└── EDA_Morango_San_Andreas.ipynb
```

## Pipeline reproduzível

O notebook `pipeline_geracao_morango.ipynb` executa, em ordem:

1. agregação das observações meteorológicas horárias;
2. construção transparente da série climática diária;
3. geração do cenário NFT experimental;
4. geração do cenário em solo convencional;
5. validação e exportação dos datasets e relatórios.

Os resultados são reproduzíveis com seed 42. NFT e solo recebem a mesma série climática e têm o mesmo número de cenários.

## Critério dos arquivos

- `clima/dados-temp.csv`: recorte meteorológico horário original.
- `clima/clima_processado.csv`: 365 dias; a coluna `origem_clima` separa observação agregada de sazonalidade sintética.
- `nft/dataset_nft.csv`: NFT experimental, com CE, nutrientes, temperatura da solução, oxigênio dissolvido, vazão e reservatório.
- `solo/dataset_solo.csv`: solo convencional, com NPK em `mg/dm³`, pH e umidade do solo.
- `semi_hidro/agrolab_morango_dataset.csv`: conjunto anterior em substrato fertirrigado; permanece separado do NFT.

## Distinção importante

`semi_hidro` e `nft` não são sinônimos. O primeiro representa cultivo em substrato; o segundo representa filme de solução nutritiva circulante e foi documentado como cenário experimental.

As bases são sintéticas e servem à validação computacional. Não demonstram superioridade produtiva sem ensaio de campo comparativo.
