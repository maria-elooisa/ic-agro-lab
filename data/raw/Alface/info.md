# AgroLab AI — Framework de Geração de Dados Sintéticos para Alface Crespa

> **Cultura-alvo:** Alface Crespa (*Lactuca sativa* var. *crispa*)
> **Escopo:** cultivo convencional em solo **e** cultivo hidropônico NFT (*Nutrient Film Technique*)
> **Finalidade:** fundamentar a construção de um dataset sintético, estatisticamente consistente e agronomicamente plausível, para treinar o classificador de ações do AgroLab AI.
> **Status:** documento de método (a geração efetiva do dataset ocorre em etapa posterior).

---

## Resumo em linguagem simples (leia isto primeiro)

O **AgroLab AI** quer demonstrar que uma IA pode tomar decisões de manejo da alface crespa melhores que as de um humano. Para treinar essa IA, precisamos de um **dataset** que ligue o **estado do cultivo** (temperatura, umidade, luz, nutrientes, pH…) à **ação correta**: não fazer nada, travar/corrigir excesso, irrigar/corrigir falta, ou proteger de extremos. Como ainda não há sensores coletando esses dados (o problema de *cold-start*), nós os **geramos de forma sintética — mas ancorados na realidade**.

O caminho, em sete passos:

1. **Validamos os números da alface** (pH, temperatura, luz, nutrientes) contra a literatura científica. → seções 2–4
2. **Baixamos o clima real** das regiões produtoras (NASA POWER) e **corrigimos seus vieses**. → seção 5
3. **Definimos as 4 ações** e a regra que decide qual tomar em cada estado. → seção 6
4. **Geramos os datasets** (solo convencional e hidropônico NFT), cruzando o clima real com estados de manejo plausíveis e rotulando cada linha com a ação certa. → seção 7
5. **Registramos a procedência** de cada coluna (medida, derivada ou estimada). → seção 8
6. **Validamos** se o dataset sintético é fiel e útil. → seção 9
7. **Registramos limitações e fontes.** → seções 10–11

Os **dois sistemas** estão gerados: solo convencional (200.900 amostras) e hidroponia NFT (200.900 amostras), ambos ancorados no mesmo clima real.

---

## 1. Introdução e Justificativa: o problema de *Cold-Start*

O AgroLab AI precisa de um conjunto de dados rotulado que associe **estados de cultivo** (leituras de sensores) a **ações de manejo** (as 4 classes do projeto). O obstáculo central é clássico em IA aplicada à agricultura: o **problema de cold-start**. Antes de instalar sensores em campo e acumular meses (ou ciclos) de séries históricas, não existe dado real suficiente para treinar um modelo supervisionado. Construir esse histórico organicamente é lento (cada ciclo de alface leva de 25 a 60 dias), caro e, pior, produz pouquíssimos exemplos das situações que mais importam: os **eventos extremos** (geada, onda de calor, falha de bomba, salinização da solução), que são raros por definição, mas são exatamente os momentos em que a decisão correta tem maior valor.

A geração de dados sintéticos resolve esse impasse permitindo:

1. **Partir do zero com cobertura ampla** — gerar um espectro completo de estados de cultivo (do ideal ao crítico) sem esperar que ocorram naturalmente.
2. **Superamostrar a cauda da distribuição** — produzir deliberadamente os casos-limite raros, equilibrando as 4 classes de ação (que, em dados reais, seriam fortemente desbalanceadas em favor da classe "não fazer nada").
3. **Controlar a causalidade** — como os dados são gerados a partir de regras agronômicas explícitas, cada rótulo de ação é rastreável à condição que o justifica, o que facilita auditoria e depuração do modelo.

A premissa metodológica é que **dados sintéticos não substituem dados reais — eles dão a partida (*bootstrap*)**. O modelo treinado em dados sintéticos serve de ponto de partida; conforme sensores reais entram em operação, o dataset é progressivamente enriquecido e o modelo, re-treinado (estratégia de *sim-to-real* / *transfer learning*).

**Nota importante sobre o realismo:** o dataset sintético não parte de faixas arbitrárias. Ele é **ancorado em séries meteorológicas históricas reais** das regiões produtoras de alface do Cinturão Verde paulista (§5). Isso é o que distingue um dataset sintético *plausível* de um *inventado*.

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

> **Origem dos dados por feature:** as features **5, 7 e 8** (temperatura, umidade relativa, radiação) são **ambientais** e serão populadas a partir das séries climáticas reais (§5). As features **1–4 e 6** (N, P, K, pH, CE/umidade) são de **manejo e sistema** — vêm das regras agronômicas (§4) e, futuramente, dos sensores da bancada NFT. O clima é o pano de fundo; o manejo é aquilo sobre o que a IA decide.

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

> **VPD sai de graça:** a série climática (§5) fornece ponto de orvalho, o que permite **calcular o VPD** sem instrumentação adicional. Ou seja, uma das variáveis que faltavam nos documentos originais já entra no dataset desde a primeira versão.

### 2.3 As 4 classes de ação (rótulo de saída)

| Classe | Ação | Gatilho agronômico (resumo) |
|--------|------|------------------------------|
| **0** | Não fazer nada | Todas as variáveis dentro da faixa ideal |
| **1** | Travar irrigação / corrigir **excesso** | CE alta, umidade saturada, excesso de N/P/K, risco de salinização |
| **2** | Irrigar / corrigir **escassez** | Umidade baixa, CE baixa, déficit hídrico ou nutricional |
| **3** | Proteger / **extremos climáticos** | Temperatura fora da faixa, radiação extrema, risco de pendoamento/*tipburn*/geada |

A lógica completa que mapeia estado → classe está em **§6**.

---

## 3. Framework Metodológico: três abordagens complementares

> **Em resumo:** para gerar dados sintéticos bons, combinamos três técnicas que se completam — um modelo que respeita a física do clima e da planta (Camada 1), a criação proposital de cenários extremos e raros (Camada 2) e a geração em massa de amostras plausíveis com correlações realistas (Camada 3).

Nenhuma técnica isolada gera um dataset agronomicamente fiel **e** estatisticamente rico. O framework combina três camadas, cada uma cobrindo uma fraqueza da outra.

### 3.1 Camada 1 — Modelos Mecanísticos (*Process-Based Crop Models*)

**O que é.** Modelos de cultivo baseados em processos fisiológicos (fotossíntese, balanço hídrico, partição de biomassa, fenologia) que, dado clima + solo + manejo, simulam o desenvolvimento da planta dia a dia. O exemplo canônico é o **DSSAT** (*Decision Support System for Agrotechnology Transfer*; Jones et al., 2003) e o **APSIM**.

**Papel no framework.** Gerar **trajetórias temporais coerentes** — séries em que temperatura, umidade, radiação e crescimento evoluem com correlações físicas reais (ex.: dias quentes e secos aumentam a ET e a demanda hídrica). É a camada que garante que o dataset não viole leis físicas/biológicas. **O insumo climático desta camada são as séries reais de §5** — não distribuições sintéticas arbitrárias.

**Ressalva crítica para alface.** O DSSAT **não possui um módulo nativo consolidado para alface** (seu foco histórico são grandes culturas: milho, trigo, soja). Usá-lo para *Lactuca sativa* exige (a) calibração de parâmetros de cultivar a partir de dados experimentais, ou (b) adaptação de um modelo genérico de folhosas. Esta limitação deve ser documentada: a Camada 1 fornece o **esqueleto temporal e as correlações climáticas**, mas os limiares fisiológicos específicos da alface crespa vêm das Camadas 2 e 3 e da literatura (§4).

**Alternativa pragmática.** Onde a calibração do DSSAT for inviável nesta fase, modelos mais simples cumprem o papel: balanço hídrico tipo FAO-56 (Penman-Monteith) para ET, e modelos de graus-dia (*growing degree days*, base ~4 °C para alface) para fenologia. São transparentes, exigem poucos parâmetros e são suficientes para gerar a espinha dorsal temporal.

**Desagregação intradiária.** As séries climáticas são **diárias** (com máxima e mínima). A reconstrução do ciclo dia/noite — necessária porque os limiares da alface são explicitamente diurnos (15–20 °C) e noturnos (8–12 °C) — é feita **nesta camada**, por interpolação senoidal entre T_min e T_max. Assim o download permanece leve e ganha-se resolução horária plausível.

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

**Calibração pelo clima real (§5).** Os casos-limite **climáticos** (geada, onda de calor) não são inventados: sua **frequência e magnitude são extraídas das séries históricas** dos polos produtores. A contagem de dias com T_min < 7 °C e T_max > 28 °C nas séries reais informa com que probabilidade a classe 3 deve aparecer — em vez de arbitrarmos um percentual. Os casos-limite **de sistema** (falha de bomba, salinização) continuam sendo injetados por regra, pois não têm contrapartida climática.

### 3.3 Camada 3 — Geração Tabular por Regras + GANs (*Rule-Based + CTGAN*)

**O que é.** Duas sub-técnicas que produzem o **volume** de amostras tabulares e suas **interdependências estatísticas**.

**(a) Geração baseada em regras (*rule-based*).** Os limiares agronômicos validados (§4) são codificados como distribuições: cada feature é amostrada de uma distribuição plausível (ex.: pH ~ Normal truncada em [5,5; 6,5] para NFT saudável), respeitando correlações conhecidas (ex.: CE alta tende a acompanhar concentração alta de N/K). É transparente, auditável e suficiente para gerar a massa de exemplos das classes 0–2.

**(b) Redes Adversárias Generativas Tabulares (*CTGAN*).** O **CTGAN** (*Conditional Tabular GAN*; Xu et al., 2019) é projetado especificamente para dados tabulares mistos (numéricos + categóricos) e lida bem com distribuições multimodais e desbalanceadas — características do nosso dataset. Seu papel: **aprender as correlações de alta ordem** entre features a partir de uma semente de dados (reais incipientes ou gerados pelas Camadas 1–2) e então amostrar novos registros que preservam a estrutura conjunta, indo além do que regras manuais capturam. A geração *condicional* permite pedir explicitamente "gere amostras da classe 3", reforçando o equilíbrio.

**Divisão de trabalho entre as sub-técnicas:** as **regras garantem plausibilidade agronômica** (nenhuma amostra viola a fisiologia da alface); o **CTGAN garante riqueza estatística** (correlações realistas que tornam o problema de classificação não-trivial). Usadas juntas, evitam tanto o dataset "de brinquedo" (regras simples demais) quanto o dataset "alucinado" (GAN sem âncora agronômica).

### 3.4 Pipeline integrado

```
   ┌──────────────────────────────┐   ┌──────────────────────────────────┐
   │ §5 Clima real (NASA POWER)   │   │ §4 Literatura validada           │
   │ séries diárias 2015–2025     │   │ faixas ideal/aceitável/crítica   │
   │ 2 polos do Cinturão Verde    │   │ (pH, CE, T, DLI, nutrição)       │
   │ → §5.11 CORREÇÃO DE VIÉS      │   │                                  │
   │   (quantile mapping INMET)   │   │                                  │
   └──────────────────────────────┘   └──────────────────────────────────┘
                │                                      │
                │  (features ambientais:               │ (parametriza
                │   T_ar, UR, radiação/DLI, VPD)       │  todas as camadas)
                ▼                                      ▼
   Camada 1 (Mecanística) ──► trajetórias temporais clima↔planta coerentes
                                     │                  + desagregação dia/noite
                                     ▼
   Camada 2 (Edge-Case)   ──► injeção dirigida de cenários críticos (classes 1,2,3)
                                     │                  frequência calibrada pelo clima real
                                     ▼
   Camada 3a (Regras)     ──► massa de amostras tabulares plausíveis
   Camada 3b (CTGAN)      ──► enriquecimento das correlações + balanceamento condicional
                                     │
                                     ▼
            ┌─────────────────────────────────────────────────────────┐
            │  Rotulagem por regras (§6): estado → classe de ação 0–3  │
            └─────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                  Validação do dataset sintético (§9)
```

---

## 4. Faixas-parâmetro validadas (alface crespa)

Estes são os valores que parametrizam as três camadas. Diferenciam **solo convencional** de **hidroponia NFT** sempre que divergem. Fontes em §11.

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

## 5. Ancoragem Climática em Dados Reais (Protocolo de Aquisição)

Esta seção documenta a **aquisição das séries meteorológicas históricas** que ancoram o dataset sintético. Sem ela, as faixas de temperatura, umidade e radiação seriam arbitrárias; com ela, o dataset reflete o clima que a alface **de fato** enfrenta nas regiões onde é mais produzida.

### 5.0 Em linguagem simples: o que foi feito nesta etapa

Antes de gerar qualquer dado sintético, precisávamos saber que clima a alface realmente enfrenta nas regiões onde ela mais é plantada. Em cinco passos:

1. **Escolhemos as regiões.** Os polos produtores do Cinturão Verde paulista: **Ibiúna + Piedade** (polo *Sudoeste*) e **Mogi das Cruzes** (polo *Alto Tietê*).
2. **Baixamos o clima real** de 2015 a 2025 da **NASA POWER** — base de satélite, gratuita e aberta, com temperatura, umidade, radiação solar, chuva e vento para qualquer coordenada.
3. **Descobrimos um problema:** os dados vinham ~**2 a 3 °C quentes demais**, porque a grade da NASA é larga (~50 km) e achatou a altitude de Ibiúna (que é alta e fria). Sem corrigir, o modelo temeria um calor que não existe e ignoraria as geadas que existem.
4. **Corrigimos o desvio** comparando mês a mês com as médias climáticas publicadas das cidades. Depois: ~18,5 °C no Sudoeste e ~19,4 °C no Alto Tietê — batendo com a realidade.
5. **Fechamos um dataset limpo:** arquivo diário (2015–2025), dois polos, com temperatura (média/máx/mín), umidade, luz (DLI), VPD e chuva — e a contagem de dias de calor e geada, que define a ação "proteger" (classe 3).

Tudo roda hoje num **único script** (`pipeline_clima.py`) que produz **dois arquivos**: o dataset final (`clima_corrigido.csv`) e um relatório (`relatorio_clima.txt`). As seções seguintes detalham cada passo.

### 5.1 Objetivo

Obter séries climáticas reais das regiões produtoras do **Cinturão Verde paulista** para parametrizar as Camadas 1 e 2 (§3.1, §3.2). O produto **não é o dataset final** — é o *molde estatístico* de onde a geração sintética amostrará os cenários.

### 5.2 Fonte de dados

**NASA POWER — API diária** (*Prediction of Worldwide Energy Resources*, NASA Langley Research Center).

| Aspecto | Descrição |
|---|---|
| Origem | Radiação derivada de satélite (CERES / GEWEX SRB); meteorologia do modelo de assimilação **MERRA-2** (NASA/GMAO) |
| Natureza | **Reanálise/satélite** — *não* é estação meteorológica de superfície (declarar no método) |
| Licença | Uso aberto, **CC BY 4.0**; sem chave de API nem cadastro |
| Resolução | ~0,5° × 0,625° (meteorologia); 1° × 1° (radiação) — células de aprox. **50 × 60 km** |
| Cobertura | 01/01/1981 até *near real time* |
| Citação | Agradecimento ao NASA LaRC POWER Project (Earth Science / Applied Science Program) — obrigatório no artigo |

### 5.3 Escopo espacial — **2 polos** (não 3 cidades)

A resolução da grade **não distingue Piedade de Ibiúna** (~22 km entre si, abaixo do tamanho da célula). Adotamos, portanto, **dois pontos climáticos**:

| Polo | Municípios | Coordenada de referência | Altitude | Köppen |
|------|-----------|--------------------------|----------|--------|
| **Sudoeste** | Ibiúna + Piedade | −23,6583 / −47,2113 (Ibiúna) | ~800–1.000 m | Cfb |
| **Alto Tietê** | Mogi das Cruzes | −23,5217 / −46,1860 | ~780 m | Cfa |

*(Coordenada complementar, para o teste de equivalência: Piedade −23,7170 / −47,4142.)*

**Justificativa da fusão.** As duas cidades são climaticamente equivalentes (mesma faixa de altitude, mesmo regime subtropical de altitude, ambas nas encostas da Serra de Paranapiacaba) e caem na mesma célula da grade. Tratá-las como séries independentes configuraria **pseudorreplicação** — duplicar o mesmo dado fingindo que são amostras distintas.

**Validação empírica obrigatória.** O script baixará as **três** coordenadas e comparará as séries de Ibiúna e Piedade. Se retornarem idênticas, a fusão fica **comprovada** (evidência citável, não suposição). Se divergirem, a decisão é revista.

### 5.4 Escopo temporal

**01/01/2015 a 31/12/2025** — 11 anos completos (4.018 dias por ponto).

Justificativa: (a) cobre 11 verões e 11 invernos, capturando variabilidade interanual e anos atípicos — essenciais para os *edge cases*; (b) período inteiramente **fora da janela de dados preliminares (NRT)**, portanto de qualidade climática consolidada; (c) evita truncar um ano pela metade.

### 5.5 Parâmetros e mapeamento para as *features*

| Parâmetro POWER | Grandeza | Uso no AgroLab |
|---|---|---|
| `T2M`, `T2M_MAX`, `T2M_MIN` | Temperatura do ar (média/máx/mín) | Feature **temperatura**; máx/mín preservam a distinção dia/noite |
| `RH2M` | Umidade relativa | Feature **umidade relativa** |
| `ALLSKY_SFC_SW_DWN` | Radiação solar incidente | Feature **radiação** → convertida em **DLI** |
| `ALLSKY_SFC_PAR_TOT` | PAR (luz fotossintética) | **DLI** direto, se disponível |
| `T2MDEW` | Ponto de orvalho | Cálculo de **VPD** |
| `PRECTOTCORR` | Precipitação | Contexto (chuva → UR alta → risco fúngico) |
| `WS2M` | Velocidade do vento | Evapotranspiração (FAO-56) |

### 5.6 Variáveis derivadas

- **DLI** (mol/m²·dia), a partir da radiação. O script **lê as unidades retornadas pela própria API** em vez de assumi-las, aplicando a conversão adequada (fração PAR ≈ 0,45 e fator ≈ 4,57 µmol/J, quando partindo da radiação de onda curta). Confronto direto com a faixa-alvo de §4.4 (**12–16 mol/m²·dia**).
- **VPD** (kPa), a partir de temperatura e ponto de orvalho (equação de Tetens). Faixa-alvo: **0,8–1,2 kPa** (§2.2).

> O **DLI é, por definição, uma grandeza diária** (mol/m²·**dia**). Para a feature de radiação, portanto, o dado diário não é aproximação — é a granularidade correta.

### 5.7 Granularidade: diário agora, horário na geração

Adota-se o dado **diário com máxima e mínima**. A desagregação intradiária (reconstrução do ciclo dia/noite por interpolação senoidal) ocorre na **Camada 1** (§3.1), onde pertence conceitualmente. Isso mantém o download leve sem perder a distinção entre limiares diurnos (15–20 °C) e noturnos (8–12 °C).

### 5.8 Controle de qualidade

- Tratamento do valor sentinela de ausência (`-999`) — **jamais** convertê-lo em zero.
- Verificação de continuidade da série (4.018 dias, sem lacunas).
- Sanidade física: T_min ≤ T_méd ≤ T_max; UR ∈ [0, 100] %; radiação ≥ 0.
- Confronto das médias obtidas com a climatologia publicada (Ibiúna ~18 °C; Mogi das Cruzes ~19,5 °C) — divergência grande indica erro na requisição.

### 5.9 Saídas (pipeline unificado)

Todo o processo — coleta, controle de qualidade, fusão dos polos, cálculo de DLI/VPD e correção de viés — roda em **um único script**, `pipeline_clima.py`, que substitui os antigos `coleta_clima_nasa_power.py` e `corrige_vies_clima.py`. Ele gera apenas **dois arquivos**:

1. `clima_corrigido.csv` — o **dataset climático final** (diário, 2 polos, 2015–2025), já corrigido e com DLI e VPD calculados. É este que a geração sintética consome.
2. `relatorio_clima.txt` — **relatório único** com unidades da API, QC, teste de equivalência Ibiúna×Piedade, correção de viés (antes→depois) e a **frequência real da classe 3**.

Os `clima_bruto_{cidade}.csv` ficam apenas como **cache/rastreabilidade** do download original: se existirem, o script os reaproveita e não consulta a NASA de novo.

> **Organização em pasta.** Todas as saídas são gravadas numa subpasta **`clima/`** (criada automaticamente pelo script), mantendo a raiz do projeto limpa: `clima/clima_corrigido.csv`, `clima/relatorio_clima.txt` e os `clima/clima_bruto_*.csv`. Os arquivos de referência INMET (opcionais, rota 2) ficam na pasta do script.

### 5.10 Perfil climático esperado (climatologia de referência)

| Polo | Média anual | Mês + frio (jul) | Mês + quente (fev) | Geada | Chuva/ano |
|------|-------------|------------------|--------------------|-------|-----------|
| Sudoeste (Ibiúna/Piedade) | ~18–18,7 °C | ~13 °C | ~22 °C | **Sim** (outono–inverno) | ~1.400–1.650 mm |
| Alto Tietê (Mogi) | ~19,5–20 °C | ~15 °C | ~23 °C | Ocasional | ~1.300–1.540 mm |

**Leitura agronômica:** as duas regiões vivem, na maior parte do ano, **dentro ou logo abaixo da faixa ideal da alface (15–20 °C)** — o que confirma por que são polos produtores e explica por que a **classe 0 (não fazer nada) será naturalmente dominante**. Os extremos moram nas bordas do calendário: o **verão** (médias de 22–23 °C, picos acima de 28–30 °C) é o bolso de risco de **pendoamento**; o **inverno**, com geadas, é o bolso de risco de **necrose por frio**. Ambos alimentam a **classe 3**, com frequências que a série real — e não um palpite — vai determinar.

### 5.11 Correção de viés (NASA POWER → referência de superfície)

A coleta revelou um **viés quente sistemático** na série NASA POWER, que precisa ser corrigido **antes** da geração sintética — caso contrário, todo o dataset herda um clima 2–3 °C mais quente que o real, e o modelo aprende a proteger contra um calor que não existe e a **ignorar o frio que existe**.

**Diagnóstico.** A grade de ~50 km usa a *altitude média da célula*, não a do município. Ibiúna (996 m) foi lida como se estivesse a ~570 m, gerando **+2,8 °C na média anual e +3,6 °C em julho**. O viés é **sazonal**: maior no inverno, porque o resfriamento noturno e o acúmulo de ar frio em vales são fenômenos locais que uma grade grossa não resolve. Consequência crítica: os eventos de geada (classe 3 por frio) ficaram **severamente subestimados**. Mogi das Cruzes (780 m) teve viés menor (+1,1 °C) e dentro da tolerância.

**Abordagem adotada (rota 2, com rota 1 como *fallback*):**

| Rota | Método | Quando usar |
|------|--------|-------------|
| **2 (principal)** | *Quantile mapping* empírico, mês a mês, contra uma **estação INMET real**. Corrige a média **e a forma da distribuição** (inclusive as caudas frias). | Sempre que houver série INMET próxima disponível (BDMEP). |
| **1 (*fallback*)** | *Delta sazonal*: desloca cada mês pela diferença (normal climatológica − média POWER). Transparente, sem *download* extra. | Quando não há estação INMET, ou como referência cruzada. |

**Ressalva de altitude (declarar no artigo).** A estação INMET mais próxima do polo Sudoeste (Sorocaba, ~600 m) é **mais baixa** que Ibiúna (996 m) — logo, mais quente. Usá-la crua *sub-corrige* o frio. Por isso o procedimento aplica um **ajuste de *lapse rate*** (6,5 °C/km) à referência, trazendo-a para a altitude-alvo antes do mapeamento. Para Mogi, uma estação de altitude semelhante dispensa o ajuste.

**Fonte das normais (rota 1).** Ibiúna, Piedade e Mogi das Cruzes **não possuem estação INMET própria** com série de 30 anos, então não há "normal oficial" INMET por município. A melhor fonte mensal por cidade é a **climatologia modelada do climate-data.org (WorldClim, 1991–2021)**: Ibiúna com média anual **18,5 °C** (jan 21,1 / jul 15,3) e Mogi das Cruzes com **19,5 °C** (jan 22,2 / jul 16,0). São valores *modelados*, não medidos — citáveis como referência, mas por isso a rota 2 (estação INMET real + *lapse rate*) permanece o padrão-ouro.

**Validação.** Após a correção pela rota 1, a média anual corrigida passa a **coincidir com a climatologia publicada** (Sudoeste **20,8 → 18,5 °C**; Alto Tietê **20,6 → 19,4 °C**). A classe 3 total cai de ~40 % (inflada pelo viés + gatilho de pico) para **~6–7 %**, com o frio concentrado no inverno (maio–setembro). **Ressalva honesta:** como o climate-data.org pode ele próprio subestimar o frio de altitude, a frequência de geada da rota 1 (Sudoeste ~3,5 %) é provavelmente *conservadora* — a rota 2, com estação real trazida à altitude de Ibiúna, tende a revelar mais dias frios. Por isso a rota 2 é a definitiva.

**Ordem no *pipeline*:** `coleta` → **`correção de viés`** → geração sintética. A correção entra entre a §5 (aquisição) e a §3 (framework), e é pré-requisito da rotulagem (§6).

---

## 6. Lógica de rotulagem das 4 classes (estado → ação)

> **Em resumo:** dado um estado do cultivo, decidimos a ação certa checando os riscos em ordem de gravidade — primeiro extremo climático (proteger), depois excesso (travar), depois falta (irrigar/adubar); se nada ocorre, não fazer nada. É essa regra que gera o rótulo de cada linha do dataset.

A rotulagem é **baseada em regras de prioridade**: avalia-se primeiro o risco mais grave (extremos climáticos), depois excesso, depois escassez; se nada dispara, a ação é "não fazer nada". A ordem evita ambiguidade quando múltiplas condições coexistem.

```python
def rotular_acao(estado, sistema):  # sistema ∈ {"solo", "nft"}
    # --- PRIORIDADE 1: extremos climáticos → Classe 3 (Proteger) ---
    # Calor: distingue CRÔNICO (pendoamento, processo acumulativo) de AGUDO (pico).
    calor_cronico = estado.T_media_movel_3d > 24   # média sustentada ≥3 dias
    calor_agudo   = estado.T_max > 32              # pico severo pontual (raro)
    frio          = estado.T_min < 7               # necrose marginal / geada
    if (frio or calor_cronico or calor_agudo
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

- Os limiares vêm diretamente de **§4** e mudam conforme a **fase fenológica** (a CE-alvo cresce do berçário ao crescimento pleno). O dataset deve carregar a fase como contexto.
- **Gatilho de calor refinado (importante):** uma versão anterior usava o *pico diário* (`T_max > 28 °C`) como proxy de estresse, o que **superestimava** a classe 3 (chegava a ~40 % dos dias — implausível para uma região onde a alface é viável). Um dia que toca 28 °C às 14 h e cai a 18 °C à noite **não** é uma emergência. O pendoamento é induzido por **temperatura média sustentada**, não por um pico isolado. Por isso o gatilho separa **calor crônico** (média móvel de 3 dias > 24 °C — o verdadeiro sinal de pendoamento) de **calor agudo** (pico > 32 °C — estresse pontual raro). Só essa mudança derruba a classe 3 de ~40 % para **~15–19 %** (antes da correção de viés) e para **~9–10 %** (depois).
- **pH fora da faixa** pode ser tratado como sub-caso de correção (associado às classes 1/2 conforme a direção do desvio) ou como uma quinta condição, a depender da granularidade desejada do projeto.
- A separação **solo × NFT** é essencial: a mesma "umidade alta" significa coisas diferentes (encharcamento no solo vs CE/diluição em NFT) e leva a ações distintas.
- O *tipburn* é um caso sutil: pode exigir ação (classe 3 ou ajuste de Ca/VPD) **mesmo com nutrição geral adequada** — um bom motivo para incluir VPD, temperatura da solução e cálcio no vetor estendido.
- Quando aplicada às séries reais de §5, esta função produz a **distribuição natural das classes** — a linha de base contra a qual o balanceamento sintético (§3.2) será calibrado.

---

## 7. Geração dos Datasets (implementação)

> Esta seção cobre os **dois sistemas**: o dataset de **solo convencional** (§7.1–7.7) e o de **hidroponia NFT** (§7.8). A procedência de cada coluna — medida, derivada ou *proxy* — está na §8.

### 7.1 Em linguagem simples

Cada linha do dataset é **um dia de clima real + um cenário de manejo + a ação correta**. O gerador percorre os 11 anos de clima corrigido (§5) e, para cada dia, cria vários cenários de manejo diferentes (mais seco, mais adubado, pH fora, etc.), calculando para cada um qual seria a ação certa segundo as regras da §6. O resultado é um conjunto grande e equilibrado, com a explicação (`motivo`) de cada decisão.

### 7.2 Como cada amostra é montada

- **Ambiente** (temperatura, umidade, luz/DLI, VPD): vem direto do `clima/clima_corrigido.csv` — clima real ancorado (§5).
- **Manejo** (N, P, K, pH, umidade do solo): sorteado por **situação** (ideal / déficit / excesso / seco / encharcado / pH fora), o que dá controle sobre o equilíbrio das classes.
- **Cultivar**: sorteada entre cinco cultivares crespas; as tolerantes ao calor pendoam a temperaturas mais altas (deslocam o gatilho da classe 3).
- **Fase fenológica**: `dias_apos_transplante` (1–45) e a `fase` derivada (muda → crescimento → desenvolvimento → colheita).
- **Rótulo**: a ação (0–3) sai das regras de prioridade da §6.

### 7.3 Variáveis do dataset (schema alinhado ao documento V2, Tabela 11)

| Grupo | Colunas |
|-------|---------|
| Contexto | `timestamp`, `polo`, `sistema`, `cultivar`, `tolerancia_calor`, `dias_apos_transplante`, `fase` |
| Manejo (decisão) | `N`, `P`, `K`, `ph`, `umidade_solo_pct` |
| Ambiente (clima real) | `temp_ar_c`, `temp_max_c`, `temp_min_c`, `temp_solo_c`, `umidade_relativa_pct`, `vpd_kpa`, `dli_mol_m2_d` |
| Rótulo | `classe_acao` (0–3), `motivo` |
| Alvos derivados | `nitrato_mg_kg`, `saude_pct` |

### 7.4 Parâmetros e fontes

- **pH do solo**: ideal 6,0–6,8; crítico <5,5 ou >7,5 (§4.1 / V2 Tab. 1).
- **P e K**: classes do **IAC Boletim 100** (Raij et al., 1996), grupo hortaliças — P déficit <26 / excesso >120 mg/dm³; K déficit <60 / excesso >235 mg/dm³.
- **Cultivares**: tolerantes ('Vera', 'Verônica', 'Vanda') pendoam com média móvel de 3 dias >26–27 °C; padrão ('Grand Rapids', 'Simpson') >24 °C (V2 §2.3).
- **Extremos climáticos**: geada (T_min <7 °C), calor agudo (T_max >32 °C), calor crônico (média de 3 dias acima do limiar da cultivar).

### 7.5 Resultado (primeira geração)

- **200.900 amostras**, 23 colunas.
- Distribuição das classes: **0 = 35,8% · 1 = 26,2% · 2 = 31,9% · 3 = 6,2%** — classe 0 majoritária (realista), classes 1 e 2 bem povoadas, classe 3 na frequência natural do clima.
- O efeito da cultivar é visível: cultivares tolerantes pendoam ~15× menos que as padrão.

### 7.6 O que é proxy e o que ficou para depois (honestidade)

- **N em mg/dm³** é um *proxy* (o Boletim 100 não interpreta N por análise de solo; N é manejado por dose). A validar.
- **`nitrato_mg_kg`** e **`saude_pct`** são derivados por proxy, não medidos.
- **`biomassa_g`** e **`num_folhas`** (alvos de regressão do V2) **não** são gerados: exigem modelo de crescimento, e não se fabrica dado sem base. Ficam para etapa dedicada.

### 7.7 Organização dos arquivos

O gerador `gerar_dataset_solo.py` lê `clima/clima_corrigido.csv` e grava na subpasta **`solo/`**: o `dataset_solo.csv` e o `relatorio_dataset_solo.txt`. A versão hidropônica (NFT) seguirá o mesmo padrão em pasta própria.

---

### 7.8 Dataset hidropônico (NFT)

**Em linguagem simples.** Mesma lógica do solo, mas agora a planta vive numa solução nutritiva em circulação. Isso muda o que medimos e o que pode dar errado: em vez de "umidade do solo", olhamos a **CE da solução**; e entram problemas que só existem aqui — solução quente demais, falta de oxigênio na água, reservatório vazio.

**O que muda em relação ao solo (e por quê):**

| Mudança | Detalhe | Ganho |
|---|---|---|
| **N deixa de ser proxy** | N, P, K, Ca, Mg em **mg/L**, pela solução de Furlani/IAC (V2 Tab. 4) | Resolve a fragilidade do dataset de solo: aqui a concentração é medida e controlada |
| **CE no lugar da umidade** | Alvo **muda com a fase**: muda 0,5–0,8 → pleno 1,2–2,0 mS/cm (V2 Tab. 2) | Sinal rico: CE 0,7 é adequada na muda, mas é déficit no crescimento pleno |
| **Nutrientes escalam com a CE** | CE é proxy da concentração iônica total; N ≈ 196 × (CE/2,0) | Cria a correlação CE↔N,K pedida em §3.3 — validado: em CE≈2,0 o N deu 192 mg/L (Furlani: 196) |
| **Variáveis novas do NFT** | `temp_solucao_c` (18–24 °C), `od_mg_l` (>5), `nivel_reservatorio_pct` | Cobrem as variáveis críticas apontadas no V2 §2.8 |
| **O₂ derivado da física** | OD = saturação(temperatura) × aeração | Reproduz a relação real "quanto mais quente, menos O₂" (V2 Tab. 7) |
| **Ambiente protegido** | Estufa: T_max +1,5 °C, T_min +1,5 °C, UR +5 pp, luz ×0,70 | Hidroponia comercial é sob cobertura; o clima externo chega **atenuado** |

**Novos gatilhos de classe 3 (exclusivos do NFT):** solução acima de 27 °C (derruba o O₂) e **hipóxia radicular** (OD < 4 mg/L, tipicamente por falha de bomba). São *falhas de sistema* — eventos raros, com probabilidade baixa no gerador para não inflar a classe artificialmente.

**Resultado:** **200.900 amostras**, 30 colunas. Distribuição: **0 = 39,2% · 1 = 15,9% · 2 = 25,4% · 3 = 19,5%**.

> **Por que a classe 3 é maior no NFT (19,5%) que no solo (6,2%)?** Não é erro — é agronomia. O solo tem **capacidade de tamponamento**: guarda água e nutrientes, e absorve oscilações. O NFT não: a planta depende de um fluxo contínuo, e qualquer falha (bomba, aquecimento da solução) vira emergência imediata. O V2 registra exatamente isso. O dataset, portanto, ensina ao modelo que **o sistema hidropônico exige vigilância maior** — o que é uma conclusão válida para o artigo.

**Organização:** `gerar_dataset_nft.py` lê `clima/clima_corrigido.csv` e grava em **`nft/`** (`dataset_nft.csv` + `relatorio_dataset_nft.txt`). Os dois datasets compartilham a coluna `sistema`, permitindo concatená-los num conjunto único.

---

## 8. Procedência dos dados: medido, derivado e *proxy*

### 8.1 O que é um *proxy* (e por que isso importa)

Um **proxy** é um substituto: uma variável usada *no lugar* de outra que não se pode medir diretamente, porque as duas andam juntas. Para estimar a riqueza de um bairro, ninguém abre a conta bancária dos moradores — olha-se consumo de energia, tipo de carro, valor dos imóveis. Nenhum desses *é* a riqueza, mas todos se correlacionam com ela.

No dataset, "proxy" significa: **este número não foi medido nem retirado diretamente de uma fonte científica — foi estimado a partir de outra grandeza**. Registrar isso importa por três razões práticas:

1. **Defesa do trabalho.** Perguntado "de onde veio esse número?", é preciso responder se a coluna é validada ou estimada. Marcar a diferença é o que separa um trabalho honesto de um que aparenta uma precisão que não tem.
2. **Interpretação do modelo.** Se o classificador apontar `saude_pct` como variável mais importante, isso **não é descoberta** — é circularidade, pois `saude_pct` foi construída a partir da própria classe de ação. O mesmo vale para qualquer proxy derivado do rótulo.
3. **Prioridade de melhoria.** Os proxies marcam exatamente onde dados reais (bancada NFT, análise de solo, orientação técnica) mudariam o dataset para melhor.

### 8.2 Os três níveis de procedência

| Nível | Significado | Como tratar no artigo |
|-------|-------------|------------------------|
| **Ancorado** | Vem de dado real ou de faixa validada na literatura | Citar a fonte normalmente |
| **Derivado (fundamentado)** | Calculado por lei física ou relação consolidada | Citar declarando a equação/relação usada |
| **Proxy (estimado)** | Construído por suposição plausível, sem fonte direta | **Declarar explicitamente como estimativa** |

### 8.3 Procedência coluna a coluna

**Ancorado — dado real ou literatura validada**

| Coluna | Origem |
|--------|--------|
| `temp_ar_c`, `temp_max_c`, `temp_min_c`, `umidade_relativa_pct`, `dli_mol_m2_d`, `temp_ar_externa_c` | Clima real NASA POWER corrigido (§5) |
| `ph` | Faixas validadas: solo 6,0–6,8; NFT 5,5–6,5 (§4.1 / V2 Tab. 1) |
| `N`, `P`, `K`, `Ca`, `Mg` (**NFT**) | Solução de Furlani/IAC em mg/L (§4.6 / V2 Tab. 4) |
| `P`, `K` (**solo**) | Classes do IAC Boletim 100, grupo hortaliças |
| `ce_ms_cm`, `ce_alvo_min`, `ce_alvo_max` | Faixas por fase (§4.2 / V2 Tab. 2) |
| `temp_solucao_c` (faixa), `od_mg_l` (limiares) | V2 Tab. 7 |
| `cultivar`, `tolerancia_calor` | Tolerância varietal documentada (V2 §2.3) |
| `classe_acao`, `motivo` | Regras de prioridade da §6, a partir das faixas validadas |
| `timestamp`, `polo` | Data e polo produtor da série climática real (§5) |
| `sistema` | Rótulo do sistema de cultivo (`solo` / `hidroponia_nft`) |

**Derivado (fundamentado) — calculado por relação física consolidada**

| Coluna | Relação usada |
|--------|----------------|
| `vpd_kpa` | Equação de Tetens, a partir de temperatura e umidade |
| `od_mg_l` (**NFT**) | Solubilidade do O₂ em função da temperatura × fator de aeração |
| `N`, `P`, `K`, `Ca`, `Mg` (**NFT**) escalando com a CE | CE é medida da concentração iônica total — prática padrão da hidroponia. **Validado:** em CE ≈ 2,0 o N resultou em 192 mg/L, contra 196 de Furlani |
| Ambiente interno da estufa (**NFT**) | Atenuação do clima externo por fatores declarados (T +1,5 °C, UR +5 pp, luz ×0,70) |

**Proxy (estimado) — declarar como estimativa**

| Coluna | Como foi construído | Fragilidade |
|--------|---------------------|-------------|
| **`N` (solo)** | Limiares de déficit (<15) e excesso (>50 mg/dm³) assumidos | **A mais frágil.** O Boletim 100 **não interpreta N por análise de solo** — o N mineral varia demais com chuva, temperatura e mineralização, e na prática é manejado por **dose (kg/ha)**, não por teor. Some no NFT, onde o N é concentração real da solução |
| `temp_solo_c` | Média móvel de 3 dias do ar + ruído (o solo amortece a variação) | Direção correta; magnitude depende de solo e cobertura |
| `nitrato_mg_kg` | "Mais N → mais nitrato; mais luz → menos nitrato" (V2 §2.8) | A relação é real; **os coeficientes são assumidos**, não de estudo |
| `saude_pct` | Score penalizando desvios das faixas ideais | **Circular:** derivado da própria classe de ação. Não usar como evidência de importância de variável |
| `umidade_solo_pct` | % da água disponível; irrigar <50%, encharcado >100% | Prática padrão para raiz rasa; limiares refináveis |
| `nivel_reservatorio_pct` (**NFT**) | Sorteado; repor abaixo de 30% | Limiar operacional assumido, não medido |
| `dias_apos_transplante`, `fase` | Sorteados uniformemente (1–45 dias) | Não seguem coorte real de plantio |

### 8.4 O que deliberadamente **não** foi gerado

`biomassa_g` e `num_folhas` — alvos de regressão previstos no schema do V2 — **não constam** nos datasets. Gerá-los exigiria um modelo de crescimento calibrado para alface crespa; produzi-los por suposição seria fabricar dado sem base. Ficam para etapa dedicada, caso o projeto avance para predição de produtividade.

### 8.5 Nota de método para o artigo

> As variáveis ambientais derivam de séries climáticas reais (NASA POWER, 2015–2025) corrigidas por viés; as faixas agronômicas seguem literatura validada (IAC/Furlani, IAC Boletim 100, Embrapa, SciELO); as variáveis de manejo são sintetizadas dentro dessas faixas. As colunas assinaladas como *proxy* na §8.3 constituem estimativas construídas para viabilizar o treinamento inicial e devem ser substituídas por medições assim que houver instrumentação.

---

## 9. Protocolo de validação do dataset sintético

Gerar dados não basta; é preciso provar que são **fiéis** e **úteis**. Quatro testes mínimos:

1. **Fidelidade estatística (univariada e bivariada).** Comparar distribuições de cada feature e correlações par-a-par entre o sintético e os dados reais disponíveis — incluindo, para as features ambientais, **as séries climáticas de §5**. Testes de Kolmogorov–Smirnov por variável e matriz de correlação. As distribuições sintéticas devem cair dentro das faixas de §4 e reproduzir correlações conhecidas (ex.: CE↔N, T_ar↔ET).

2. **Plausibilidade agronômica (sanity checks por regra).** Nenhuma amostra pode violar restrições físicas/biológicas: pH ∈ faixa viável, sem T_ar = 50 °C, sem CE negativa, fase fenológica coerente com idade da planta. Amostras que violem são descartadas ou corrigidas.

3. **Utilidade para ML — TSTR (*Train on Synthetic, Test on Real*).** O teste decisivo: treinar o classificador **só** no sintético e avaliá-lo em dados **reais** (mesmo que poucos). Se o desempenho em real for próximo do desempenho treino-real/teste-real, o sintético é útil. (A métrica complementar TRTS — treinar no real, testar no sintético — ajuda a detectar lacunas de cobertura.)

4. **Cobertura de casos-limite e balanceamento.** Verificar que as 4 classes estão representadas em proporção adequada (não necessariamente 25 % cada, mas com a cauda suficientemente povoada) e que cada cenário crítico de §3.2 aparece em volume treinável.

> **Sobre números de desempenho de ML "prontos":** afirmações como "redes neurais reduzem o MSE em 73,88 %" ou "Random Forest atinge 94,45 % de acurácia", presentes nos documentos de origem, **não são rastreáveis a um estudo verificável** e não devem ser citadas como metas. Métricas-alvo devem emergir do próprio protocolo TSTR neste projeto.

---

## 10. Limitações e ressalvas

**Do framework de geração**

- **DSSAT/APSIM sem módulo nativo de alface:** a Camada 1 exige calibração ou substituição por modelos mais simples (graus-dia, FAO-56). Documentar a escolha feita.
- **Dependência de cultivar:** limiares de pendoamento, *tipburn* e ciclo variam fortemente entre cultivares de alface crespa. Idealmente o dataset registra a cultivar; na ausência, usar faixas conservadoras e sinalizar a incerteza.
- **Sintético ≠ real:** o dataset sintético é *bootstrap*, não verdade final. O compromisso é re-treinar com dados reais à medida que a instrumentação avança (*sim-to-real*).
- **Conversões e escalas:** sempre declarar a escala de CE→ppm e a base de pH (água vs CaCl₂) para evitar erros sistemáticos.

**Dos dados climáticos (§5)**

- **Reanálise, não medição local:** os dados NASA POWER derivam de satélite e modelo de assimilação. Não capturam microclima de estufa, ilhas de calor urbanas nem efeitos de encosta. **Parcialmente mitigado** pela correção de viés (§5.11): o *quantile mapping* contra estação INMET real reancora a distribuição na medição de superfície, e o ajuste de *lapse rate* recupera o efeito de altitude que a grade perde. O viés residual (microclima de estufa vs campo aberto) permanece e deve ser tratado como fator de atenuação.
- **A grade não resolve Piedade × Ibiúna:** fusão documentada e comprovada empiricamente (§5.3), não presumida.
- **Coordenadas são centroides municipais**, não das áreas rurais de produção efetiva.
- **Cobertura parcial das features:** o clima popula **3 das 8 features** (temperatura, umidade relativa, radiação) — as demais (N, P, K, pH, CE) são de manejo/sistema.
- **Ambiente aberto vs protegido:** as séries refletem condições de campo aberto. Um sistema NFT em ambiente protegido (como a bancada do projeto) tem temperatura e UR **amortecidas** em relação ao clima externo — o que deve ser modelado como um fator de atenuação, não ignorado.

---

## 11. Referências acadêmicas

**Modelagem e geração de dados sintéticos**

- **JONES, J. W. et al.** The DSSAT cropping system model. *European Journal of Agronomy*, v. 18, n. 3–4, p. 235–265, 2003. DOI: [10.1016/S1161-0301(02)00107-7](https://doi.org/10.1016/S1161-0301(02)00107-7) — modelo mecanístico de cultivo (Camada 1).
- **XU, L. et al.** Modeling Tabular Data using Conditional GAN (CTGAN). *Advances in Neural Information Processing Systems (NeurIPS)*, v. 32, 2019. [arXiv:1907.00503](https://arxiv.org/abs/1907.00503) · implementação: [github.com/sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN) — geração tabular sintética (Camada 3b).
- **SHORTEN, C.; KHOSHGOFTAAR, T. M.** A survey on image data augmentation for deep learning. *Journal of Big Data*, v. 6, n. 60, 2019. *(Atenção: a referência "Shorten 2021" citada nos documentos originais trata de aumento de dados de **imagem/texto**, não tabular; para dados tabulares a referência pertinente é Xu et al. 2019.)*

**Dados climáticos**

- **NASA POWER.** *Prediction of Worldwide Energy Resources* — NASA Langley Research Center, Earth Science / Applied Science Program. Portal: [power.larc.nasa.gov](https://power.larc.nasa.gov/) · Documentação da API diária: [power.larc.nasa.gov/docs/services/api/temporal/daily](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) · Dicionário de parâmetros: [power.larc.nasa.gov/docs/tutorials/parameters](https://power.larc.nasa.gov/docs/tutorials/parameters/) · Origem dos dados (CERES, MERRA-2): [power.larc.nasa.gov/docs/faqs/data](https://power.larc.nasa.gov/docs/faqs/data/) · Licença CC BY 4.0: [registry.opendata.aws/nasa-power](https://registry.opendata.aws/nasa-power/)
- **NASA EARTHDATA.** Política de uso e citação de dados. [earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance)
- **CLIMATE-DATA.ORG.** Climatologia mensal modelada (WorldClim, 1991–2021) — normais de referência da rota 1 (correção de viés). Ibiúna: [ibiuna-34791](https://en.climate-data.org/south-america/brazil/sao-paulo/ibiuna-34791/) · Mogi das Cruzes: [mogi-das-cruzes-4112](https://en.climate-data.org/south-america/brazil/sao-paulo/mogi-das-cruzes-4112/). *Climatologia modelada, não medição de estação; a rota 2 (INMET/BDMEP) é a referência rigorosa.*
- **INMET / BDMEP.** Banco de Dados Meteorológicos para Ensino e Pesquisa — séries de estação para a correção de viés (rota 2). [bdmep.inmet.gov.br](https://bdmep.inmet.gov.br/) · Normais Climatológicas 1991–2020: [portal.inmet.gov.br/normais](https://portal.inmet.gov.br/normais)

> **Agradecimento a incluir no artigo:** conforme solicitado pelo projeto, deve-se creditar a obtenção dos dados ao *NASA Langley Research Center (LaRC) POWER Project*, financiado pelo *NASA Earth Science / Applied Science Program*. O texto exato está na documentação oficial.

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

*Documento de método do Projeto AgroLab AI — iniciação científica. Adaptado para alface crespa (Lactuca sativa var. crispa) a partir do framework original (concebido para morango). As faixas-parâmetro e referências foram validadas contra literatura primária e fontes oficiais; afirmações sem fonte rastreável foram explicitamente sinalizadas e descartadas. A ancoragem climática (§5) usa dados abertos da NASA POWER para as regiões produtoras do Cinturão Verde paulista.*