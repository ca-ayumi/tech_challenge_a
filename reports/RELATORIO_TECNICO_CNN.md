# Relatório técnico — CNN em mamografias (CBIS-DDSM local)

Documento de apoio ao notebook `notebooks/cnn/02_cnn_mammography.ipynb`, descrevendo análise exploratória, pré-processamento, arquitetura, treino e leitura dos resultados.

---

## 1. Contexto e dados

O notebook classifica **achados** como **benigno (0)** ou **maligno (1)** a partir de **imagens JPEG recortadas** (`cnn_data/jpeg/`), alinhadas aos metadados do CBIS-DDSM via CSVs em `cnn_data/csv/` (`dicom_info.csv`, descrições de massa/calcificação em treino e teste, coluna de patologia, etc.). Os recortes reduzem custo computacional e focam a região de interesse, com o trade-off de **perder contexto da mamografia completa**.

---

## 2. Discussões da análise exploratória

- **Integração CSV ↔ arquivos.** Caminhos do *dataset* original são convertidos para o layout local (`cnn_data/jpeg/...`). Arquivos `*:Zone.Identifier` são ignorados por não corresponderem a JPEGs reais.

- **Conjuntos treino / validação / teste.** Os CSVs já definem **treino** e **teste**. Uma fatia de **validação** é obtida **apenas do treino** (`train_test_split` estratificado), evitando vazamento de informação do conjunto de teste para ajuste de hiperparâmetros ou *threshold*.

- **Distribuição de classes.** Após o *split*, o notebook mostra contagens por conjunto. Em **modo rápido** (`QUICK_MODE = True`), aplica-se **amostragem balanceada por classe** com limites máximos por classe (treino, validação e teste), para treinar em CPU com tempo previsível.

- **Inspeção visual.** Amostras de imagens por classe confirmam leitura correta dos ficheiros e dão intuição sobre variabilidade de brilho/contraste.

- **Probabilidades e separação.** Na avaliação, a distribuição das probabilidades preditas na validação ajuda a ver se o modelo **separa** benignos e malignos ou se as curvas estão muito sobrepostas (sinal de fraco aprendizado ou *threshold* inadequado).

---

## 3. Estratégias de pré-processamento

| Etapa | Descrição |
|--------|-----------|
| **Rótulos** | Normalização de strings em `pathology` / metadados para mapear benigno/maligno de forma consistente. |
| **Resolução** | Redimensionamento para **128×128** pixels. |
| **Canal** | Imagens em escala de cinzentos (**1 canal**), adequado a mamografia em tons de cinza. |
| **Normalização numérica** | Valores no intervalo **[0, 1]**; o carregamento pode **padronizar contraste por imagem** (mamografias têm brilho muito variável, o que estabiliza o treino na CNN). |
| **Treino com TensorFlow** | `tf.data`: leitura sob demanda, *batch*, *prefetch* e paralelismo (`AUTOTUNE`) para throughput em CPU. |
| **Pesos de classe** | `compute_class_weight('balanced')` convertido em dicionário passado a `model.fit`, compensando desbalanceamento sem depender só do *sampling*. |

**Modo rápido (`QUICK_MODE`):** amostragem balanceada com tetos por classe (valores no código: por exemplo até **900** treino/classe, **220** validação/classe, **276** teste/classe — sujeitos ao que existir após filtros). Para treinar com mais dados, aumentar limites ou definir `QUICK_MODE = False`.

---

## 4. Modelo usado e porquê

### 4.1 Arquitetura (Keras 3 / TensorFlow 2.16)

- **Entrada:** tensor `128 × 128 × 1`.
- **Aumento de dados (*data augmentation*):** espelhamento horizontal, rotação leve e *zoom* suave — aumenta robustez sem exigir GPU.
- **Blocos convolucionais:** três estágios com **32**, **64** e **128** filtros; cada estágio tem duas convoluções **3×3**, `BatchNormalization`, `ReLU`, `MaxPooling2D` e `Dropout` crescente (0.10 → 0.15 → 0.20) para regularização.
- **Cabeça de classificação:** `GlobalAveragePooling2D` (menos parâmetros que `Flatten`, menos risco de overfitting em base pequena), `Dense(96, relu)`, `Dropout(0.35)`, `Dense(1, sigmoid)` — saída é **probabilidade de maligno**.

**Porquê esta família de escolhas:** arquitetura propositalmente **simples e didática**, executável em **CPU**; `BatchNorm` estabiliza otimização; *dropout* e *pooling* global mitigam overfitting quando a amostra efetiva é limitada.

### 4.2 Treino

| Hiperparâmetro | Valor típico (notebook) |
|----------------|-------------------------|
| `BATCH_SIZE` | 16 |
| `EPOCHS` (máximo) | 12 |
| Otimizador | `Adam`, *learning rate* inicial **5e-4** |
| Função de perda | `binary_crossentropy` |
| Métricas | Acurácia binária, **AUC**, precisão, *recall* |

**Callbacks:** `EarlyStopping` monitorizando **`val_auc`**, modo `max`, *patience* 4, `restore_best_weights=True`; `ReduceLROnPlateau` também em `val_auc` (*factor* 0.5, *patience* 2, `min_lr=1e-5`). Ou seja, o desempenho de **ranking** (AUC) guia tanto a paragem antecipada como o ajuste fino da taxa de aprendizagem.

### 4.3 Inferência e métricas no teste

- Sobre as probabilidades da validação, escolhe-se o **melhor limiar (*threshold*)** para a métrica em estudo (alinhado ao foco em *recall* do maligno).
- Aplica-se o mesmo limiar no **teste**; reportam-se matriz de confusão e métricas. O *recall* do maligno é destacado como métrica central (**falsos negativos** atrasam investigação clínica).

---

## 5. Resultados e interpretação

- **Curvas de histórico:** acurácia e AUC em treino e validação (e *recall* quando disponível no histórico) permitem discutir **ajuste** versus **subajuste** e comparar com `EarlyStopping`.

- **Distribuição das probabilidades na validação:** se as curvas de benignos e malignos estiverem muito sobrepostas, o modelo ainda não aprendeu separação visual suficiente — útil para interpretação qualitativa além de um único número de AUC.

- **Matriz de confusão no teste** (com o *threshold* escolhido na validação): traduz decisões operacionais em contagens de VP/VN/FP/FN.

- **Artefatos em `reports/`** (após execução completa): por exemplo `cnn_sample_images.png`, curvas de treino, figuras de probabilidades e matriz de confusão, conforme células do notebook.

**Interpretação de contexto:** a CNN opera em **recortes**, não na mamografia inteira; o modo rápido favorece **portabilidade e ensino**, não um modelo clínico pronto para produção.

---

## 6. Limitações

- Dependência de **dados locais** em `cnn_data/jpeg/` (não versionados por tamanho).
- **Generalização** a outros centros, aparelhos ou populações não é garantida.
- **Uso clínico:** apenas apoio conceitual à triagem; decisão médica permanece com profissionais de saúde.

---

*Gerado como documentação do repositório Tech Challenge — fase CNN.*
