# AgroLab AI — Geração de Dados Sintéticos e Simulação Avançada da Cultura do Morango
**Projeto de Iniciação Científica**

## 1. Introdução e Justificativa Metodológica

Um desafio extremamente comum na agricultura inteligente (*Smart Farming*) e no aprendizado de máquina aplicado a IoT é o problema de *Cold-Start* (início frio) e a severa escassez de dados. Coletar um conjunto de dados robusto, confiável e bem balanceado ao longo de múltiplos ciclos de cultivo (como o do morango) pode levar vários meses ou até anos. Além disso, repositórios públicos raramente contêm todas as sete variáveis ambientais e de solo sincronizadas em um mesmo histórico.

Para superar essa limitação e viabilizar o desenvolvimento do núcleo preditivo do jogo de decisão 2D do AgroLab, este notebook utiliza a metodologia de **Synthetic Data Generation (Geração de Dados Sintéticos)**.

Abaixo, detalhamos as abordagens consagradas na literatura científica que validam, legitimam e fundamentam a estratégia utilizada neste projeto.

---

## Methodological Framework (Fundamentação Metodológica)

### 1. Mechanistic Models (Modelagem Mecanística baseada em Regras Físicas)
* **O que é:** Esta metodologia consiste em gerar dados aplicando equações matemáticas e físicas que regem a natureza real, em vez de gerar números puramente aleatórios. Na agricultura, simuladores internacionais renomados (como *APSIM* e *DSSAT*) utilizam este princípio.
* **Por que estamos utilizando:** O morangueiro possui limites biológicos estritos documentados cientificamente (como a faixa ideal de pH entre 5.5 e 6.2, e estresse térmico acima de 30°C). Nosso algoritmo utiliza equações de correlação e decaimento: por exemplo, modelamos a evapotranspiração real, onde o aumento da temperatura ambiente força um decaimento matemático na umidade do solo. Isso garante que a base de dados gerada respeite as leis da física e da biologia vegetal.

### 2. Edge-Case Simulation (Simulação de Cenários Raros / Casos de Borda)
* **O que é:** É a técnica de induzir estatisticamente a ocorrência de eventos extremos ou anomalias dentro do ecossistema de dados para fins de treinamento de modelos preditivos.
* **Por que estamos utilizando:** Em uma estufa real e bem gerenciada, condições de toxicidade crítica por excesso de NPK ou geadas severas (< 5°C) são raríssimas. Se usássemos apenas dados de um cenário real estável, nosso modelo sofreria de *Class Imbalance* (Desbalanceamento de Classes) e nunca aprenderia quando acionar os comandos críticos de **`1` (Travar Irrigação)** ou **`3` (Proteger)**. A simulação nos permite estressar o sistema de forma segura, criando um modelo de IA muito mais resiliente e preparado para crises.

### 3. Rule-Based Tabular Generation (Geração Tabular Baseada em Regras de Negócio)
* **O que é:** Uma abordagem de engenharia de dados onde a rotulagem (*labeling*) das classes alvo é determinada por restrições lógicas e condicionais rígidas baseadas no conhecimento de especialistas humanos (*Domain Expert Rules*).
* **Por que estamos utilizando:** Como o núcleo do AgroLab AI é um jogo focado em demonstrar a superioridade da IA contra a tomada de decisão humana, as ações recomendadas pelo modelo precisam ser 100% acuradas em relação ao manual de manejo do morango (`morango_agrolab.pdf`). Mapear essas regras diretamente no gerador garante que o modelo *baseline* (como uma árvore de decisão ou Random Forest) aprenda perfeitamente a fronteira de decisão que o jogador humano tentará enfrentar no *frontend*.

---

## Academic References for Citation (Referências para Citação Acadêmica)

Para garantir o rigor científico do relatório final desta Iniciação Científica, apoiamo-nos nos seguintes marcos teóricos:

* **JONES, J. W. et al.** *The DSSAT cropping system model.* European Journal of Agronomy, 2003. (Valida o uso de modelos matemáticos e simulações mecânicas para prever interações solo-clima-planta).
* **XU, L. et al.** *Modeling Tabular data using Conditional GAN.* NeurIPS, 2019. (Fundamenta a importância e a evolução da geração de dados sintéticos tabulares de alta fidelidade para o treinamento de Machine Learning).
* **SHORTEN, C. et al.** *A survey on Synthetic Data Generation.* arXiv, 2021. (Discute como a simulação de dados resolve o problema de falta de dados e o "Cold-Start" em sistemas de sensores IoT).