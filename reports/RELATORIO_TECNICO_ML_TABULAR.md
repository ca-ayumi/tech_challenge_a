# Relatório técnico — Machine Learning tabular (câncer de mama)

Documento de apoio ao notebook `notebooks/tabular/01_ml_breast_cancer.ipynb`, descrevendo análise exploratória, pré-processamento, modelos e leitura dos resultados.

---

## 1. Contexto e dados

O fluxo utiliza o arquivo `breast_cancer_wisconsin.csv` na raiz do repositório: **569** observações com medidas numéricas de núcleo celular (raio, textura, perímetro, área, suavidade, compactação, concavidade, simetria, dimensão fractal, etc., em versões *mean*, *se* e *worst*). O alvo é `diagnosis` (**B** = benigno, **M** = maligno), mapeado internamente para **0** e **1**, sendo **1** (maligno) a classe positiva por representar o maior risco clínico em triagem.

---

## 2. Discussões da análise exploratória

- **Distribuição da classe.** O notebook examina contagens de benignos e malignos (gráfico de barras) para entender o desbalanceamento. Isso orienta o uso de `class_weight='balanced'` e a escolha de métricas sensíveis a falsos negativos.

- **Valores ausentes.** Há verificação de `NaN` por coluna; no conjunto típico do Wisconsin costuma não haver lacunas relevantes, o que simplifica o pré-processamento.

- **Comparação entre classes.** São calculadas médias por classe (`groupby('diagnosis')`) e a diferença absoluta entre médias de benignos e malignos, destacando quais atributos numéricos mais separam as classes.

- **Correlação com o diagnóstico.** É calculada a matriz de correlação; variáveis com maior correlação linear (em valor absoluto) com `diagnosis` aparecem no topo. Um *heatmap* restringido às variáveis mais correlacionadas melhora a legibilidade e evidencia blocos de covariância (muitas medidas derivadas do mesmo núcleo tendem a correlacionar entre si).

- **Distribuições univariadas.** *Facets* (*seaborn.FacetGrid*) das principais variáveis correlacionadas mostram sobreposição ou separação visual entre benigno e maligno, útil para intuir separabilidade linear versus necessidade de modelos mais flexíveis.

---

## 3. Estratégias de pré-processamento

| Etapa | Descrição |
|--------|-----------|
| Remoção de colunas vazias | `dropna(axis=1, how='all')` elimina colunas totalmente vazias (por exemplo `Unnamed: 32` em algumas exportações do CSV). |
| Remoção de `id` | O identificador não é característica clínica do tumor; retirá-lo evita *data leakage* espúrio ou peso indevido no modelo. |
| Codificação do alvo | `B` → 0, `M` → 1. |
| Partição treino/teste | `train_test_split` com `test_size=0.2`, `stratify=y` e `random_state=42` para preservar a proporção de classes e reprodutibilidade. |

Modelos sensíveis à escala (**Regressão logística**) usam `Pipeline` com `StandardScaler` antes do estimador. A **árvore de decisão** e a **random forest** não exigem normalização para ordem de *splits*, mas convivem no mesmo *grid* de experimentação com a mesma matriz de features.

---

## 4. Modelos usados e porquê

| Modelo | Configuração relevante | Justificativa |
|--------|-------------------------|----------------|
| **Regressão logística** | `Pipeline(StandardScaler + LogisticRegression)`, `max_iter=2000`, `class_weight='balanced'` | Baseline forte em problemas quase lineares; coeficientes interpretáveis; escalonamento necessário pelas magnitudes heterogêneas das features. |
| **Árvore de decisão** | `max_depth=4`, `class_weight='balanced'` | Modelo de baixa complexidade, fácil de inspecionar (*splits* explícitos); profundidade limitada reduz overfitting no Wisconsin. |
| **Random Forest** | `n_estimators=300`, `max_depth=None`, `class_weight='balanced'`, `n_jobs=-1` | Captura não linearidades e interações entre variáveis correlacionadas; ensemble reduz variância em relação a uma única árvore; oferece **importância de variáveis** nativa. |

**Critério de seleção do “melhor” modelo no notebook:** ordenação por **recall da classe maligno** e **F1 da classe maligno** (com desempate implícito na estrutura do `DataFrame` de resultados). Isso alinha o experimento à triagem: **minimizar falsos negativos** costuma ser prioritário, aceitando-se eventualmente mais falsos positivos.

---

## 5. Resultados e interpretação

- **Métricas reportadas** por modelo (no conjunto de teste): acurácia, precisão do maligno, *recall* do maligno e F1 do maligno, além de `classification_report` e **matriz de confusão** para o modelo escolhido.

- **Interpretação qualitativa (saídas típicas após executar o notebook).** A regressão logística costuma apresentar bom equilíbrio entre métricas no Wisconsin; a random forest pode elevar precisão do maligno com *recall* ligeiramente menor, conforme limiar interno de folhas. A matriz de confusão e a mensagem de “interpretação rápida” enfatizam que **falso negativo** = maligno classificado como benigno — o erro mais sensível no contexto de triagem.

- **Explicabilidade.** Para a *Random Forest* vencedora no *ensemble* de explicabilidade: gráfico de importância de variáveis; tentativa de **SHAP** em amostra pequena (para custo computacional) com gráfico de impacto na classe maligna. O SHAP pode falhar em alguns ambientes; o notebook trata exceção e segue.

- **Artefatos gravados em `reports/`** (quando o notebook é executado até o fim): por exemplo `tabular_correlation_heatmap.png`, `tabular_confusion_matrix.png`, `tabular_feature_importance.png`, etc.

---

## 6. Limitações (discussão crítica do notebook)

- Os modelos são **demonstrações acadêmicas**; não substituem avaliação médica nem decisão clínica.
- **Validações futuras** deveriam incluir dados externos, monitoramento de **viéses** e revisão clínica dos erros.
- O *dataset* Wisconsin é clássico e **não representa** toda a diversidade populacional ou de aquisição de imagens/exames reais.

---

*Gerado como documentação do repositório Tech Challenge — fase tabular.*
