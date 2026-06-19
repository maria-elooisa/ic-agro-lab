# AgroLab AI — Framework de Geração de Dados Sintéticos para Alface Crespa

> **Cultura-alvo:** Alface Crespa (*Lactuca sativa* var. *crispa*)
> **Escopo:** cultivo convencional em solo **e** cultivo hidropônico NFT (*Nutrient Film Technique*)
> **Finalidade:** fundamentar a construção de um dataset sintético, estatisticamente consistente e agronomicamente plausível, para treinar o classificador de ações do AgroLab AI.
> **Status:** documento de método (a geração efetiva do dataset ocorre em etapa posterior).

---

## 1. Introdução e Justificativa: o problema de *Cold-Start*

O AgroLab AI precisa de um conjunto de dados rotulado que associe **estados de cultivo** (leituras de sensores) a **ações de manejo** (as 4 classes do projeto). O obstáculo central é clássico em IA aplicada à agricultura: o **problema de cold-start**. Antes de instalar sensores em campo e acumular meses (ou ciclos) de séries históricas, não existe dado real suficiente para treinar um modelo supervisionado. Construir esse histórico organicamente é lento (cada ciclo de alface leva de 25 a 60 dias), caro e, pior, produz pouquíssimos exemplos das situações que mais importam: os **eventos extremos** (geada, onda de calor, falha de bomba, salinização da solução), que são raros por definição, mas são exatamente os momentos em que a decisão correta tem maior valor.

A geração de dados sintéticos resolve esse impasse permitindo:

1. **Partir do zero com cobertura ampla** — gerar um espectro completo de estados de cultivo (do ideal ao crítico) sem esperar que ocorram naturalmente.
2. **Superamostrar a cauda da distribuição** — produzir deliberadamente os casos-limite raros, equilibrando as 4 classes de ação (que, em dados reais, seriam fortemente desbalanceadas em favor da classe "não fazer nada").
3. **Controlar a causalidade** — como os dados são gerados a partir de regras agronômicas explícitas, cada rótulo de ação é rastreável à condição que o justifica, o que facilita auditoria e depuração do modelo.

A premissa metodológica é que **dados sintéticos não substituem dados reais — eles dão a partida (*bootstrap*)**. O modelo treinado em dados sintéticos serve de ponto de partida; conforme sensores reais entram em operação, o dataset é progressivamente enriquecido e o modelo, re-treinado (estratégia de *sim-to-real* / *transfer learning*).

---

## 2. Escopo: variáveis-alvo e classes de ação

### 2.1 Vetor de *features* (estado de entrada)

O classificador recebe, a cada instante de decisão, um vetor de estado. O núcleo definido pelo projeto:

| # | Feature | Símbolo | Convencional (solo) | Hidropônico (NFT) |
|---|---------|---------|---------------------|-------------------|
| 1 | Nitrogênio | N | mg/dm³ no solo / kg/ha | mg/L na solução |
| 2 | Fósforo | P | mg/dm³ no solo / kg/ha | mg/L na solução |
| 3 | Potássio | K | mg/dm³ no solo / kg/ha | mg/L na solução |
| 4 | pH | pH | pH do solo (água/CaCl₂) | pH da solução |
| 5 | Temperatura do ar | T_ar | °C | °C |
| 6 | Umidade do meio | θ | umidade do solo (% v/v) | **CE da solução (mS/cm)** + umidade do substrato na muda |
| 7 | Umidade relativa do ar | UR | % | % |
| 8 | Radiação / luz | Rad | W/m² ou DLI (mol/m²·dia) | DLI (mol/m²·dia) ou PPFD (µmol/m²·s) |

> **Nota de consistência:** a descrição original do projeto menciona "7 features" mas enumera 8 grandezas. Recomenda-se tratar **radiação/DLI** como feature de pleno direito (é determinante para a alface — ver §4.4), totalizando 8. A grande diferença entre sistemas está na **feature 6**: em solo mede-se a *umidade do solo*; em NFT a variável análoga e mais informativa é a *condutividade elétrica (CE)* da solução, pois a planta vive em fluxo contínuo de solução e não há "solo" para reter água.

### 2.2 Features adicionais recomendadas (forward-looking)

A análise crítica dos documentos (entregável complementar) identificou variáveis ausentes que **elevam substancialmente o realismo** e devem ser incorporadas assim que houver instrumentação. O framework já as prevê como colunas opcionais do dataset sintético:

| Feature | Por que importa para alface | Faixa de referência |
|---------|-----------------------------|---------------------|
| Oxigênio dissolvido (OD) | Raiz em NFT sofre hipóxia; governa absorção | >5 mg/L (ideal ~8 mg/L a 25 °C) |
| Temperatura da solução | Controla OD e velocidade de absorção; gatilho de *tipburn* | 18–24 °C (>27 °C crítico) |
| VPD (déficit de pressão de vapor) | Liga T_ar e UR; rege transpiração e *tipburn* | 0,8–1,2 kPa |
| Evapotranspiração (ET) | Dimensiona a reposição de água/solução | ~75–100 mL/planta·dia |
| CO₂ | Limita fotossíntese em ambiente fechado | 400–1200 ppm |
| Nitrato foliar (NO₃⁻) | Qualidade e limite legal (UE) — ver §4.6 | conforme época/sistema |
| Temperatura do solo | Em campo, afeta germinação e absorção | — |

### 2.3 As 4 classes de ação (rótulo de saída)

| Classe | Ação | Gatilho agronômico (resumo) |
|--------|------|------------------------------|
| **0** | Não fazer nada | Todas as variáveis dentro da faixa ideal |
| **1** | Travar irrigação / corrigir **excesso** | CE alta, umidade saturada, excesso de N/P/K, risco de salinização |
| **2** | Irrigar / corrigir **escassez** | Umidade baixa, CE baixa, déficit hídrico ou nutricional |
| **3** | Proteger / **extremos climáticos** | Temperatura fora da faixa, radiação extrema, risco de pendoamento/*tipburn*/geada |

A lógica completa que mapeia estado → classe está em **§5**.

---

## 3. Framework Metodológico: três abordagens complementares

Nenhuma técnica isolada gera um dataset agronomicamente fiel **e** estatisticamente rico. O framework combina três camadas, cada uma cobrindo uma fraqueza da outra.

### 3.1 Camada 1 — Modelos Mecanísticos (*Process-Based Crop Models*)

**O que é.** Modelos de cultivo baseados em processos fisiológicos (fotossíntese, balanço hídrico, partição de biomassa, fenologia) que, dado clima + solo + manejo, simulam o desenvolvimento da planta dia a dia. O exemplo canônico é o **DSSAT** (*Decision Support System for Agrotechnology Transfer*; Jones et al., 2003) e o **APSIM**.

**Papel no framework.** Gerar **trajetórias temporais coerentes** — séries em que temperatura, umidade, radiação e crescimento evoluem com correlações físicas reais (ex.: dias quentes e secos aumentam a ET e a demanda hídrica). É a camada que garante que o dataset não viole leis físicas/biológicas.

**Ressalva crítica para alface.** O DSSAT **não possui um módulo nativo consolidado para alface** (seu foco histórico são grandes culturas: milho, trigo, soja). Usá-lo para *Lactuca sativa* exige (a) calibração de parâmetros de cultivar a partir de dados experimentais, ou (b) adaptação de um modelo genérico de folhosas. Esta limitação deve ser documentada: a Camada 1 fornece o **esqueleto temporal e as correlações climáticas**, mas os limiares fisiológicos específicos da alface crespa vêm das Camadas 2 e 3 e da literatura (§4).

**Alternativa pragmática.** Onde a calibração do DSSAT for inviável nesta fase, modelos mais simples cumprem o papel: balanço hídrico tipo FAO-56 (Penman-Monteith) para ET, e modelos de graus-dia (*growing degree days*, base ~4 °C para alface) para fenologia. São transparentes, exigem poucos parâmetros e são suficientes para gerar a espinha dorsal temporal.

### 3.2 Camada 2 — Simulação de Casos-Limite (*Edge-Case Simulation*)

**O que é.** Geração dirigida (não aleatória) dos cenários raros e críticos que os dados reais quase nunca capturam, mas que definem a competência do modelo.

**Papel no framework.** Garantir **cobertura das classes 1, 2 e 3** — sobretudo a classe 3 (extremos). Em vez de esperar uma geada, injeta-se deliberadamente o cenário "T_ar = 2 °C às 5 h da manhã" e rotula-se a ação correta (proteger). Casos-limite essenciais para alface crespa:

- **Onda de calor / risco de pendoamento:** T_ar média >20–25 °C por dias consecutivos (a alface pendoa precocemente com calor + dias longos — ver §4.3).
- **Geada / necrose por frio:** T_ar <7 °C (necrose marginal) a <0 °C.
- **Salinização da solução (NFT):** CE subindo acima de 2,5–3,0 mS/cm por evaporação/superdosagem.
- **Diluição / falha de fertirrigação:** CE despencando abaixo de 0,8 mS/cm.
- **Estresse hídrico em solo:** umidade do solo abaixo do ponto de murcha.
- **Hipóxia radicular (NFT):** temperatura da solução >27 °C derrubando o OD.
- ***Tipburn* (queima de bordas):** combinação VPD alto + transpiração intensa + cálcio insuficiente, mesmo com nutrição geral adequada.

**Método.** Para cada caso-limite, define-se a região do espaço de features que o caracteriza e amostra-se densamente nela, com ruído controlado para gerar variabilidade realista. Isso equilibra ativamente o dataset, contrapondo-se ao desbalanceamento natural (em que >80 % dos instantes seriam classe 0).

### 3.3 Camada 3 — Geração Tabular por Regras + GANs (*Rule-Based + CTGAN*)

**O que é.** Duas sub-técnicas que produzem o **volume** de amostras tabulares e suas **interdependências estatísticas**.

**(a) Geração baseada em regras (*rule-based*).** Os limiares agronômicos validados (§4) são codificados como distribuições: cada feature é amostrada de uma distribuição plausível (ex.: pH ~ Normal truncada em [5,5; 6,5] para NFT saudável), respeitando correlações conhecidas (ex.: CE alta tende a acompanhar concentração alta de N/K). É transparente, auditável e suficiente para gerar a massa de exemplos das classes 0–2.

**(b) Redes Adversárias Generativas Tabulares (*CTGAN*).** O **CTGAN** (*Conditional Tabular GAN*; Xu et al., 2019) é projetado especificamente para dados tabulares mistos (numéricos + categóricos) e lida bem com distribuições multimodais e desbalanceadas — características do nosso dataset. Seu papel: **aprender as correlações de alta ordem** entre features a partir de uma semente de dados (reais incipientes ou gerados pelas Camadas 1–2) e então amostrar novos registros que preservam a estrutura conjunta, indo além do que regras manuais capturam. A geração *condicional* permite pedir explicitamente "gere amostras da classe 3", reforçando o equilíbrio.

**Divisão de trabalho entre as sub-técnicas:** as **regras garantem plausibilidade agronômica** (nenhuma amostra viola a fisiologia da alface); o **CTGAN garante riqueza estatística** (correlações realistas que tornam o problema de classificação não-trivial). Usadas juntas, evitam tanto o dataset "de brinquedo" (regras simples demais) quanto o dataset "alucinado" (GAN sem âncora agronômica).

### 3.4 Pipeline integrado

```
            ┌─────────────────────────────────────────────────────────┐
            │  Literatura validada (§4): faixas ideal/aceitável/crítica│
            └─────────────────────────────────────────────────────────┘
                                     │ (parametriza todas as camadas)
                                     ▼
   Camada 1 (Mecanística) ──► trajetórias temporais clima↔planta coerentes
                                     │
   Camada 2 (Edge-Case)   ──► injeção dirigida de cenários críticos (classes 1,2,3)
                                     │
   Camada 3a (Regras)     ──► massa de amostras tabulares plausíveis
   Camada 3b (CTGAN)      ──► enriquecimento das correlações + balanceamento condicional
                                     │
                                     ▼
            ┌─────────────────────────────────────────────────────────┐
            │  Rotulagem por regras (§5): estado → classe de ação 0–3  │
            └─────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                  Validação do dataset sintético (§6)
```

---

## 4. Faixas-parâmetro validadas (alface crespa)

Estes são os valores que parametrizam as três camadas. Diferenciam **solo convencional** de **hidroponia NFT** sempre que divergem. Fontes em §8.

### 4.1 pH

| Sistema | Ideal | Crítico |
|---------|-------|---------|
| Solo convencional | 6,0–6,8 | <5,5 ou >7,5 |
| Hidroponia NFT | 5,5–6,5 | <4,5 ou >7,5 |

**Correção de pH (importante):** para **elevar** o pH em hidroponia usa-se **hidróxido de potássio (KOH)** — *não* hidróxido de sódio (NaOH), pois o sódio não é nutriente, acumula na solução recirculante e causa toxicidade/desequilíbrio iônico; o KOH ainda repõe potássio. Para **baixar** o pH usa-se ácido fosfórico ou ácido nítrico.

### 4.2 Condutividade Elétrica (CE) — NFT

| Fase | CE (mS/cm) | ppm (escala 500) |
|------|-----------|------------------|
| Germinação / berçário | 0,5–0,8 | 250–400 |
| Crescimento inicial | 0,8–1,2 | 400–600 |
| Crescimento pleno | 1,2–2,0 | 600–1000 |

> **Declarar sempre a escala de conversão.** ppm = CE × fator, e o fator varia: **500** (NaCl, EUA), 640 (Europa), 700 (KCl, Austrália). Recomenda-se **trabalhar em mS/cm** no dataset e tratar ppm como derivado, evitando ambiguidade. A alface tolera ampla faixa de CE: estudos com a solução de Furlani mostram que ~50 % da força iônica (CE ≈ 0,98 mS/cm) ainda sustenta ~90 % da produção máxima — ou seja, **flutuações moderadas de CE não colapsam a biomassa** (a afirmação de "−8 a 12 % por ±0,4 mS/cm" presente nos documentos não tem fonte primária e não deve ser usada como regra).

### 4.3 Temperatura do ar e pendoamento

| Parâmetro | Faixa |
|-----------|-------|
| Diurna ideal | 15–20 °C |
| Noturna ideal | 8–12 °C |
| Necrose marginal (frio) | <7 °C |
| **Indução de pendoamento** | médias **>20–25 °C** (não apenas >30 °C) |
| Estresse térmico severo | >30 °C |

> **Correção de premissa:** o pendoamento (*bolting*) é induzido por **temperaturas médias acima de ~20–25 °C combinadas com dias longos**, não somente acima de 30 °C. Por isso o gatilho da **classe 3 (proteger)** por calor deve ativar já em torno de **25–28 °C**, antes do dano severo. Cultivares variam muito: há materiais tolerantes ao calor (ex.: linhas da Embrapa) que resistem mais.

### 4.4 Radiação / DLI (*Daily Light Integral*)

| Parâmetro | Valor |
|-----------|-------|
| DLI ideal (alface) | **12–16 mol/m²·dia** |
| DLI berçário/mudas | 6–9 mol/m²·dia |
| PPFD típico (indoor) | 200–300 µmol/m²·s |

> **Correção de premissa:** a alface **não** é "planta de dia longo que cresce mais com fotoperíodo longo". Dias longos somados a calor **induzem o pendoamento** (indesejável). O que de fato impulsiona o crescimento vegetativo comercial é o **DLI** (quantidade total de luz/dia), não a duração do dia em si. Acima de ~14–17 mol/m²·dia o ganho satura e pode haver *tipburn*.

### 4.5 Umidade relativa e ciclo

| Parâmetro | Valor |
|-----------|-------|
| UR ideal | 60–75 % |
| UR alta (>80–85 %) | favorece fungos (míldio, *Botrytis*) → risco fitossanitário |
| Ciclo (transplante → colheita) | ~25–45 dias (total desde semeadura pode chegar a 45–60) |

### 4.6 Nutrição

**Solução nutritiva de referência (Furlani / IAC, folhosas, força plena, mg/L):**

| Elemento | Faixa | Elemento | Faixa |
|----------|-------|----------|-------|
| N total | ~190–202 | Fe | 1,8–2,0 |
| P | ~39–47 | Mn | 0,4–0,5 |
| K | ~180–193 | B | 0,3–0,5 |
| Ca | ~142–150 | Zn | 0,06–0,2 |
| Mg | ~27–40 | Cu | 0,02–0,1 |
| S | ~46–52 | Mo | 0,06–0,1 |

Ordem de micronutrientes por concentração: **Fe > Mn > B > Zn > Cu**.

**Fósforo em solo (correção importante):** a dose de "733–756 kg/ha de P₂O₅" citada nos documentos **não é generalizável** — corresponde a solos de altíssima fixação de P. Valores documentados: máximos quadráticos de **617–672 kg/ha** (alface americana, solo de alta fixação; Mota, Yuri & Resende, 2003) ou resposta **linear até ~200 kg/ha** em Latossolo distrófico. **Regra para o dataset:** condicionar a dose de P ao tipo de solo (50–200 kg/ha como faixa típica; doses altas apenas com justificativa edáfica).

**Nitrato foliar — limites legais UE** (relevante para qualidade/exportação; Regulamento (UE) nº 1258/2011, posteriormente consolidado no Reg. (UE) 2023/915):

| Tipo de alface | Limite (mg NO₃/kg) |
|----------------|--------------------|
| Sob cobertura, colheita out–mar | 5000 |
| Ar livre, colheita out–mar | 4000 |
| Sob cobertura, colheita abr–set | 4000 |
| Ar livre, colheita abr–set | 3000 |
| Iceberg (cobertura / ar livre) | 2500 / 2000 |

### 4.7 Solução nutritiva — parâmetros físico-químicos (NFT)

| Parâmetro | Faixa |
|-----------|-------|
| Temperatura da solução | 18–24 °C (>27 °C crítico) |
| Oxigênio dissolvido | >5 mg/L (ideal ~8 mg/L) |
| ET (consumo) | ~75–100 mL/planta·dia |

---

## 5. Lógica de rotulagem das 4 classes (estado → ação)

A rotulagem é **baseada em regras de prioridade**: avalia-se primeiro o risco mais grave (extremos climáticos), depois excesso, depois escassez; se nada dispara, a ação é "não fazer nada". A ordem evita ambiguidade quando múltiplas condições coexistem.

```python
def rotular_acao(estado, sistema):  # sistema ∈ {"solo", "nft"}
    # --- PRIORIDADE 1: extremos climáticos → Classe 3 (Proteger) ---
    if (estado.T_ar < 7 or estado.T_ar > 28
            or estado.Rad_extrema
            or (sistema == "nft" and estado.T_solucao > 27)):
        return 3  # proteger: geada, onda de calor/pendoamento, radiação extrema, hipóxia

    # --- PRIORIDADE 2: excesso → Classe 1 (Travar irrigação / corrigir excesso) ---
    if sistema == "nft":
        ce_max = 2.0   # mS/cm (fase de crescimento pleno)
        if estado.CE > ce_max or estado.UR > 85 or estado.excesso_NPK:
            return 1
    else:  # solo
        if estado.umidade_solo > capacidade_campo or estado.excesso_NPK:
            return 1

    # --- PRIORIDADE 3: escassez → Classe 2 (Irrigar / corrigir escassez) ---
    if sistema == "nft":
        if estado.CE < 0.8 or estado.deficit_nutricional:
            return 2
    else:  # solo
        if estado.umidade_solo < ponto_murcha or estado.deficit_hidrico:
            return 2

    # --- Tudo dentro do ideal → Classe 0 (Não fazer nada) ---
    return 0
```

**Observações sobre a rotulagem:**

- Os limiares (7 °C, 28 °C, CE 0,8–2,0, etc.) vêm diretamente de **§4** e mudam conforme a **fase fenológica** (a CE-alvo cresce do berçário ao crescimento pleno). O dataset deve carregar a fase como contexto.
- **pH fora da faixa** pode ser tratado como sub-caso de correção (associado às classes 1/2 conforme a direção do desvio) ou como uma quinta condição, a depender da granularidade desejada do projeto.
- A separação **solo × NFT** é essencial: a mesma "umidade alta" significa coisas diferentes (encharcamento no solo vs CE/diluição em NFT) e leva a ações distintas.
- O *tipburn* é um caso sutil: pode exigir ação (classe 3 ou ajuste de Ca/VPD) **mesmo com nutrição geral adequada** — um bom motivo para incluir VPD, temperatura da solução e cálcio no vetor estendido.

---

## 6. Protocolo de validação do dataset sintético

Gerar dados não basta; é preciso provar que são **fiéis** e **úteis**. Quatro testes mínimos:

1. **Fidelidade estatística (univariada e bivariada).** Comparar distribuições de cada feature e correlações par-a-par entre o sintético e qualquer amostra real disponível (testes de Kolmogorov–Smirnov por variável; matriz de correlação). As distribuições sintéticas devem cair dentro das faixas de §4 e reproduzir correlações conhecidas (ex.: CE↔N, T_ar↔ET).

2. **Plausibilidade agronômica (sanity checks por regra).** Nenhuma amostra pode violar restrições físicas/biológicas: pH ∈ faixa viável, sem T_ar = 50 °C, sem CE negativa, fase fenológica coerente com idade da planta. Amostras que violem são descartadas ou corrigidas.

3. **Utilidade para ML — TSTR (*Train on Synthetic, Test on Real*).** O teste decisivo: treinar o classificador **só** no sintético e avaliá-lo em dados **reais** (mesmo que poucos). Se o desempenho em real for próximo do desempenho treino-real/teste-real, o sintético é útil. (A métrica complementar TRTS — treinar no real, testar no sintético — ajuda a detectar lacunas de cobertura.)

4. **Cobertura de casos-limite e balanceamento.** Verificar que as 4 classes estão representadas em proporção adequada (não necessariamente 25 % cada, mas com a cauda suficientemente povoada) e que cada cenário crítico de §3.2 aparece em volume treinável.

> **Sobre números de desempenho de ML "prontos":** afirmações como "redes neurais reduzem o MSE em 73,88 %" ou "Random Forest atinge 94,45 % de acurácia", presentes nos documentos de origem, **não são rastreáveis a um estudo verificável** e não devem ser citadas como metas. Métricas-alvo devem emergir do próprio protocolo TSTR neste projeto.

---

## 7. Limitações e ressalvas

- **DSSAT/APSIM sem módulo nativo de alface:** a Camada 1 exige calibração ou substituição por modelos mais simples (graus-dia, FAO-56). Documentar a escolha feita.
- **Dependência de cultivar:** limiares de pendoamento, *tipburn* e ciclo variam fortemente entre cultivares de alface crespa. Idealmente o dataset registra a cultivar; na ausência, usar faixas conservadoras e sinalizar a incerteza.
- **Sintético ≠ real:** o dataset sintético é *bootstrap*, não verdade final. O compromisso é re-treinar com dados reais à medida que a instrumentação avança (*sim-to-real*).
- **Conversões e escalas:** sempre declarar a escala de CE→ppm e a base de pH (água vs CaCl₂) para evitar erros sistemáticos.
- **Regiões produtoras (etapa posterior):** quando o dataset for ancorado em clima real de polos paulistas (ex.: Piedade, Ibiúna), as distribuições climáticas das Camadas 1–2 devem ser recalibradas para as séries locais (INMET / NASA POWER), substituindo as faixas genéricas usadas nesta fase.

---

## 8. Referências acadêmicas

**Modelagem e geração de dados sintéticos**

- **JONES, J. W. et al.** The DSSAT cropping system model. *European Journal of Agronomy*, v. 18, n. 3–4, p. 235–265, 2003. DOI: [10.1016/S1161-0301(02)00107-7](https://doi.org/10.1016/S1161-0301(02)00107-7) — modelo mecanístico de cultivo (Camada 1).
- **XU, L. et al.** Modeling Tabular Data using Conditional GAN (CTGAN). *Advances in Neural Information Processing Systems (NeurIPS)*, v. 32, 2019. [arXiv:1907.00503](https://arxiv.org/abs/1907.00503) · implementação: [github.com/sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN) — geração tabular sintética (Camada 3b).
- **SHORTEN, C.; KHOSHGOFTAAR, T. M.** A survey on image data augmentation for deep learning. *Journal of Big Data*, v. 6, n. 60, 2019. *(Atenção: a referência "Shorten 2021" citada nos documentos originais trata de aumento de dados de **imagem/texto**, não tabular; para dados tabulares a referência pertinente é Xu et al. 2019.)*

**Agronomia da alface — nutrição e manejo**

- **FURLANI, P. R.** *Instruções para o cultivo de hortaliças de folhas pela técnica de hidroponia – NFT*. Campinas: IAC, 1998 (Boletim Técnico 168). Solução nutritiva de referência — análise de concentração disponível em [scielo.br/j/hb/a/HsH735SySknvSv8QyBbFJXs](https://www.scielo.br/j/hb/a/HsH735SySknvSv8QyBbFJXs/?lang=pt).
- **MOTA, J. H.; YURI, J. E.; RESENDE, G. M. de et al.** Produção de alface americana em função de doses e fontes de fósforo. *Horticultura Brasileira*, v. 21, n. 4, 2003. [scielo.br/j/hb/a/wgq5n6wryMtcDb4HWbQ7H5K](https://www.scielo.br/j/hb/a/wgq5n6wryMtcDb4HWbQ7H5K/) — doses de P (corrige a dose de P₂O₅).
- **BENINNI, E. R. Y.; TAKAHASHI, H. W.; NEVES, C. S. V. J.** Manejo do cálcio em alface de cultivo hidropônico. *Horticultura Brasileira*, v. 21, n. 4, p. 605–610, 2003. [scielo.br/j/hb/a/VsSFtFQPKNNnFc5FtRyZ6nN](https://www.scielo.br/j/hb/a/VsSFtFQPKNNnFc5FtRyZ6nN/) — cálcio / *tipburn* / teores foliares.

**Luz e ambiente**

- **WEI, H. et al.** Determination of optimal daily light integral (DLI) for indoor cultivation of lettuce. *Scientific Reports (Nature)*, 2023. [nature.com/articles/s41598-023-36997-2](https://www.nature.com/articles/s41598-023-36997-2) — DLI ideal.
- Revisão sobre DLI e crescimento da alface (ótimo 12–16 mol/m²·dia): [PMC11667103](https://pmc.ncbi.nlm.nih.gov/articles/PMC11667103/).

**Qualidade e legislação**

- **COMISSÃO EUROPEIA.** Regulamento (UE) nº 1258/2011 — teores máximos de nitratos (consolidado/revogado pelo Reg. (UE) 2023/915). [eur-lex.europa.eu/eli/reg/2011/1258/oj](https://eur-lex.europa.eu/eli/reg/2011/1258/oj).

---

*Documento de método do Projeto AgroLab AI — iniciação científica. Adaptado para alface crespa (Lactuca sativa var. crispa) a partir do framework original (concebido para morango). As faixas-parâmetro e referências foram validadas contra literatura primária e fontes oficiais; afirmações sem fonte rastreável foram explicitamente sinalizadas e descartadas.*
