# AgroLab AI - Sistema Integrado de Gestão Agrícola Gamificada

## 📝 Descrição do Projeto
O **AgroLab AI** é um projeto de iniciação científica focado no desenvolvimento de um sistema integrado e gamificado. O grande objetivo da plataforma é demonstrar como a Inteligência Artificial (alimentada por dados preditivos e variáveis ambientais) pode otimizar a tomada de decisões no manejo de culturas agrícolas, superando a performance de decisões humanas tradicionais.

Através de uma interface dinâmica (estilo jogo 2D) alimentada por uma API (FastAPI), o usuário e a IA competem para gerenciar as condições de cultivo.

---

## 🎯 Objetivo Principal
Provar empiricamente que a tomada de decisão baseada em modelos de Machine Learning/IA maximiza a produtividade e previne riscos com maior eficácia do que a intuição ou a tomada de decisão humana convencional. O sistema calcula o impacto das ações na saúde, qualidade e biomassa da planta em tempo real.

---

## 🎮 As Regras do Jogo: Categorias de Decisão (Classes da IA)
O motor do jogo e as previsões do modelo de Machine Learning são baseados em **4 categorias de ações principais** que alteram o estado da cultura:

* **`0` - Não Fazer nada:** Todos os parâmetros estão estabilizados na faixa ótima. Nenhuma ação corretiva é necessária.
* **`1` - Nutrientes em Excesso (Travar Irrigação):** Níveis elevados de fertilizantes ou condutividade elétrica detectados. O sistema interrompe o fluxo de água/nutrientes para evitar toxicidade, desperdício ou bloqueio de outros elementos cruciais.
* **`2` - Nutrientes em Escassez (Irrigar):** Níveis abaixo do recomendado ou estresse hídrico detectado. Aciona a fertirrigação para repor água e macronutrientes.
* **`3` - Proteger (Sombrite / Ventilação / Cobertura):** Ativada em condições de extremos climáticos (luminosidade excessiva, calor extremo ou risco de geada) para mitigar o estresse térmico, escaldaduras ou o aparecimento de fungos e doenças.

---

## 📊 Features de Entrada (Variáveis do Modelo)
Os modelos de IA utilizam **7 variáveis contínuas cruciais** capturadas via sensores para determinar a ação correta:

1. **Macronutrientes (N, P, K):** Nitrogênio, Fósforo e Potássio. Monitorados para avaliar o crescimento vegetativo, floração e qualidade do fruto por fase fenológica.
2. **pH:** Regulador central. Fora da faixa ideal, bloqueia a absorção dos nutrientes na solução nutritiva.
3. **Temperatura do Ar:** Regula o ciclo metabólico geral e atua como o principal disparador crítico para as ações de proteção térmica.
4. **Umidade (Solo / Substrato):** Feature direta para acionar os comandos automatizados de irrigação ou drenagem.
5. **Umidade Relativa do Ar:** Influencia diretamente na taxa de transpiração da planta e se torna um gatilho de risco para a proliferação de fungos em níveis elevados.
6. **Radiação (lux / fotoperíodo):** Combustível essencial da fotossíntese, impactando diretamente no florescimento pleno e na concentração de açúcares nos frutos.

---

## 🌿 Culturas Monitoradas e Parâmetros de Referência

O sistema adapta suas regras de recompensa e punição no jogo de acordo com os limites científicos de cada cultura:

### 1. Alface (Foco: Ciclo Rápido e Hidroponia NFT)
A alface é altamente responsiva, com ciclo curto de 25 a 45 dias, tornando-a ideal para avaliar o pipeline de ML.
* **pH Ideal (NFT):** 5,5 a 6,5 (Limites críticos: < 4,5 ou > 7,5).
* **Condutividade Elétrica (CE):** 0,8 a 2,0 mS/cm (250 ppm em berçário até 1000 ppm em crescimento pleno).
* **Temperatura do Ar:** Diurna entre 15°C e 20°C. Valores > 30°C causam pendoamento precoce.
* **Umidade do Ar Ideal:** 60% a 75%.

### 2. Morango (Foco: Alta Sensibilidade e Estufa Semi-Hidropônica)
Cultura de alto valor comercial e extrema sensibilidade a variações climáticas, ativando frequentemente as ações de proteção.
* **pH Ideal (Hidropônico):** 5,5 a 6,2.
* **Concentração NPK Ideal:** N ($150\text{--}250\text{ mg/L}$), P ($30\text{--}60\text{ mg/L}$), K ($200\text{--}350\text{ mg/L}$).
* **Temperatura Ideal:** Frutificação ótima entre 18°C e 25°C. Dispara *Proteger* se < 5°C (geada) ou > 30°C (calor).
* **Umidade do Solo Ideal:** 60% a 80% (Alerta crítico: < 50% ou > 90%).
* **Luminosidade Mínima:** $200\ \mu\text{mol/m}^2/\text{s}$ para fotossíntese plena.

### 3. Tomate (Foco: Manejo Nutricional por Fases e Prevenção de Doenças/Rachaduras)
Hortaliça de altíssima relevância econômica, caracterizada por uma resposta complexa e dinâmica às 7 features do modelo. Muito propensa a estresses hídricos que danificam a integridade física do fruto.
* **pH Ideal:** 5,5 a 6,5 para sistemas hidropônicos (NFT/DWC). Alerta crítico se cair abaixo de 5,0 ou subir acima de 6,5.
* **Condutividade Elétrica (CE):** Intervalo ótimo entre 1,2 e 2,0 mS/cm na solução nutritiva.
* **Temperatura do Ar:** Desenvolvimento vegetativo e frutificação ideais entre 21°C e 28°C. Dispara *Proteger (Ventilação)* se $>38^{\circ}C$ e *Proteger (Cobertura)* se $<10^{\circ}C$ (interrompe a síntese de licopeno).
* **Umidade do Ar Ideal:** Diurna em torno de 75% e noturna em 85% em ambiente hidropônico. Alertas críticos para fungos acionados se a umidade relativa ultrapassar 85% combinada com calor.
* **Luminosidade Mínima:** Exigência de pelo menos 6 horas de luz direta por dia. Baixos níveis geram frutos ocos e reduzem o Grau Brix (açúcar).
* **Manejo Estratégico:** Apresenta alta sensibilidade à falta de Cálcio (Ca) associada à irrigação irregular, o que causa a podridão apical (mancha escura no fundo do fruto). Exige a suspensão completa da irrigação de 14 a 28 dias antes da colheita para concentrar açúcares.

---

## 🧠 Arquitetura de Machine Learning
O projeto prevê problemas de **Classificação** (para recomendar a melhor ação em tempo real no jogo) e **Regressão/Séries Temporais** (para prever variáveis-alvo complexas como produtividade final em kg/ha, score de qualidade do fruto e ocorrência de rachaduras):

### Pipeline de Desenvolvimento
1. **Coleta e Exploração:** Sensores físicos simulados ou dados via MQTT integrados a uma análise exploratória (EDA) baseada nas regras técnicas validadas.
2. **Modelagem Baseline:** Algoritmos como **Random Forest** e **XGBoost** para a classificação das 4 ações principais de tomada de decisão.
3. **Modelagem Temporal:** Redes **LSTM / GRU** para predizer a evolução diária dos parâmetros e o tempo estimado até a colheita, que varia de 90 a 100 dias após o transplante.
4. **Deploy:** Disponibilização via **FastAPI** integrada diretamente ao frontend do jogo, respondendo instantaneamente às escolhas do jogador humano.