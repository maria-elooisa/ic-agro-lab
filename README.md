# AgroLab AI - Sistema Integrado de Gestão Agrícola Gamificada

## 📝 Descrição do Projeto
O **AgroLab AI** é um projeto de iniciação científica focado no desenvolvimento de um sistema integrado e gamificado. O grande objetivo da plataforma é demonstrar como a Inteligência Artificial (alimentada por dados preditivos e variáveis ambientais) pode otimizar a tomada de decisões no manejo de culturas agrícolas, superando a performance de decisões humanas tradicionais.

Através de uma interface dinâmica (estilo jogo 2D) alimentada por uma API (FastAPI), o usuário e a IA competem para gerenciar as condições de cultivo.

---

## 🎯 Objetivo Principal
Provar empiricamente que a tomada de decisão baseada em modelos de Machine Learning/IA maximiza a produtividade e previne riscos com maior eficácia do que a intuição ou a tomada de decisão humana convencional[cite: 1]. O sistema calcula o impacto das ações na saúde e biomassa da planta em tempo real[cite: 1].

---

## 🎮 As Regras do Jogo: Categorias de Decisão (Classes da IA)
O motor do jogo e as previsões do modelo de Machine Learning são baseados em **4 categorias de ações principais** que alteram o estado da cultura[cite: 1]:

* **`0` - Condição Ideal:** Todos os parâmetros estão estabilizados na faixa ótima. Nenhuma ação corretiva é necessária[cite: 1].
* **`1` - Nutrientes em Excesso (Travar Irrigação):** Níveis elevados de fertilizantes ou condutividade elétrica detectados. O sistema interrompe o fluxo de água/nutrientes para evitar toxicidade e queima das raízes/folhas[cite: 1, 2].
* **`2` - Nutrientes em Escassez (Irrigar):** Níveis abaixo do recomendado ou estresse hídrico. Aciona a fertirrigação para repor água e macronutrientes[cite: 1, 2].
* **`3` - Proteger (Sombrite / Ventilação / Cobertura):** Ativada em condições de extremos climáticos (luminosidade excessiva, calor extremo ou geada) para mitigar estresse térmico ou aparecimento de fungos/doenças[cite: 1, 2].

---

## 📊 Features de Entrada (Variáveis do Modelo)
Os modelos de IA utilizam **7 variáveis contínuas cruciais** capturadas via sensores (como ESP32 em ambientes físicos) para determinar a ação correta[cite: 1, 2]:

1. **Macronutrientes (N, P, K):** Nitrogênio, Fósforo e Potássio. Medidos em $mg/L$ ou via condutividade elétrica (CE/TDS) para avaliar a concentração de sais na solução[cite: 1, 2].
2. **pH:** Regulador central. Fora da faixa ideal, bloqueia a absorção de nutrientes mesmo que o solo/solução esteja rico[cite: 1, 2].
3. **Temperatura do Ar:** Regula o ciclo metabólico e disparador crítico para a ação de *Proteger*[cite: 1, 2].
4. **Temperatura da Solução Nutritiva:** Crítico em sistemas hidropônicos (temperaturas altas reduzem o oxigênio dissolvido)[cite: 1].
5. **Umidade (Solo / Substrato):** Feature direta para acionar comandos de *Irrigar* ou *Travar Irrigação*[cite: 1, 2].
6. **Umidade Relativa do Ar:** Influencia na transpiração da planta e na proliferação de fungos sob extremos[cite: 1, 2].
7. **Luminosidade (lux / fotoperíodo):** O combustível da fotossíntese. Feature diferencial do AgroLab para controle de estufas e sombreamento[cite: 1, 2].

---

## 🌿 Culturas Monitoradas e Parâmetros de Referência

O sistema adapta suas regras de recompensa e punição no jogo de acordo com os limites científicos de cada cultura[cite: 1, 2]:

### 1. Alface (Foco: Ciclo Rápido e Hidroponia NFT)
A alface é altamente responsiva, com ciclo curto de 25 a 45 dias, tornando-a ideal para avaliar o pipeline de ML[cite: 1].
* **pH Ideal (NFT):** 5,5 a 6,5 (Limites críticos: < 4,5 ou > 7,5)[cite: 1].
* **Condutividade Elétrica (CE):** 0,8 a 2,0 mS/cm (250 ppm em berçário até 1000 ppm em crescimento pleno)[cite: 1].
* **Temperatura do Ar:** Diurna entre 15°C e 20°C. Valores > 30°C causam pendoamento precoce[cite: 1].
* **Umidade do Ar Ideal:** 60% a 75%[cite: 1].

### 2. Morango (Foco: Alta Sensibilidade e Estufa Semi-Hidropônica)
Cultura de alto valor comercial e extrema sensibilidade a variações climáticas, ativando frequentemente as ações de proteção.
* **pH Ideal (Hidropônico):** 5,5 a 6,2.
* **Concentração NPK Ideal:** N ($150\text{--}250\text{ mg/L}$), P ($30\text{--}60\text{ mg/L}$), K ($200\text{--}350\text{ mg/L}$)[cite: 2].
* **Temperatura Ideal:** Frutificação ótima entre 18°C e 25°C. Dispara *Proteger* se < 5°C (geada) ou > 30°C (calor)[cite: 2].
* **Umidade do Solo Ideal:** 60% a 80% (Alerta crítico: < 50% ou > 90%)[cite: 2].
* **Luminosidade Mínima:** $200\ \mu\text{mol/m}^2/\text{s}$ para fotossíntese plena[cite: 2].

### 3. [Terceira Cultura]
* *Espaço reservado para inclusão do próximo documento anexado.*

---

## 🧠 Arquitetura de Machine Learning
O projeto prevê problemas de **Classificação** (para recomendar a melhor ação em tempo real no jogo) e **Regressão/Séries Temporais** (para prever o ganho de peso fresco/biomassa e taxa de crescimento das culturas)[cite: 1]:

### Pipeline de Desenvolvimento
1. **Coleta e Exploração:** Sensores físicos simulados ou dados via MQTT integrados a uma análise exploratória (EDA)[cite: 1].
2. **Modelagem Baseline:** Algoritmos como **Random Forest** e **XGBoost** para classificação das ações (com acurácia referenciada na literatura de até 94.45%)[cite: 1].
3. **Modelagem Temporal:** Redes **LSTM / GRU** para predição de crescimento futuro baseado no histórico do jogador[cite: 1].
4. **Deploy:** Disponibilização via **FastAPI** integrada diretamente ao frontend do jogo[cite: 1].