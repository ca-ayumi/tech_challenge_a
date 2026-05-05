# Tech Challenge - Fase 1

Projeto de IA aplicado à saúde e segurança da mulher, com duas frentes:

- Machine Learning tabular para diagnóstico de câncer de mama com `breast_cancer_wisconsin.csv`.
- Visão computacional com CNN para classificação de mamografias usando os dados locais em `cnn_data/`.

Os notebooks usam apenas caminhos relativos e procuram automaticamente a raiz do projeto. Assim, podem ser abertos a partir da raiz, de `notebooks/`, ou da própria pasta do notebook.

## Estrutura

```text
.
├── breast_cancer_wisconsin.csv
├── cnn_data/
│   ├── csv/
│   └── jpeg/      # dados externos, não versionados
├── notebooks/
│   ├── tabular/
│   │   ├── 01_ml_breast_cancer.ipynb
│   │   └── requirements.txt
│   └── cnn/
│       ├── 02_cnn_mammography.ipynb
│       └── requirements.txt
└── reports/
```

## Requisitos gerais

- Python 3.10 ou 3.11 recomendado.
- `pip` atualizado.
- Sistema 64-bit.
- Memória suficiente para processar as imagens redimensionadas da CNN.

O notebook de CNN foi configurado para CPU local por padrão. Ele usa imagens `128x128`, batch pequeno, amostragem balanceada e `EarlyStopping` para reduzir custo computacional. Em computadores mais fracos, a execução pode demorar, mas não deve depender de GPU.

Os ambientes virtuais `.venv-tabular/` e `.venv-cnn/` são gerados localmente e não devem ser versionados. Para instalar todas as dependências em um único ambiente, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Setup automático nos notebooks

Cada notebook possui, nas primeiras células, um bootstrap de ambiente:

- cria o venv esperado se ele ainda não existir;
- instala o `requirements.txt` correspondente dentro desse venv;
- registra um kernel Jupyter específico para o notebook.

Na primeira execução, rode a célula de setup. Se o notebook ainda não estiver usando o venv correto, ela vai parar com uma mensagem pedindo para trocar o kernel:

- `Tech Challenge Tabular` para o notebook tabular;
- `Tech Challenge CNN` para o notebook de CNN.

Depois de trocar o kernel, execute o notebook novamente desde o início. O setup é idempotente: se o `requirements.txt` não mudou, a instalação é pulada nas próximas execuções.

## Execução do notebook tabular

Opcionalmente, você também pode criar o ambiente virtual manualmente:

```bash
python3 -m venv .venv-tabular
source .venv-tabular/bin/activate
pip install --upgrade pip
pip install -r notebooks/tabular/requirements.txt
python -m ipykernel install --user --name tech-challenge-tabular --display-name "Tech Challenge Tabular"
jupyter notebook notebooks/tabular/01_ml_breast_cancer.ipynb
```

No Windows PowerShell:

```powershell
py -3.11 -m venv .venv-tabular
.\.venv-tabular\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r notebooks\tabular\requirements.txt
python -m ipykernel install --user --name tech-challenge-tabular --display-name "Tech Challenge Tabular"
jupyter notebook notebooks\tabular\01_ml_breast_cancer.ipynb
```

## Execução do notebook CNN

Crie outro ambiente virtual separado:

```bash
python3 -m venv .venv-cnn
source .venv-cnn/bin/activate
pip install --upgrade pip
pip install -r notebooks/cnn/requirements.txt
python -m ipykernel install --user --name tech-challenge-cnn --display-name "Tech Challenge CNN"
jupyter notebook notebooks/cnn/02_cnn_mammography.ipynb
```

No Windows PowerShell:

```powershell
py -3.11 -m venv .venv-cnn
.\.venv-cnn\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r notebooks\cnn\requirements.txt
python -m ipykernel install --user --name tech-challenge-cnn --display-name "Tech Challenge CNN"
jupyter notebook notebooks\cnn\02_cnn_mammography.ipynb
```

## Dados

Os arquivos esperados são:

- `breast_cancer_wisconsin.csv`
- `cnn_data/csv/dicom_info.csv`
- `cnn_data/csv/mass_case_description_train_set.csv`
- `cnn_data/csv/mass_case_description_test_set.csv`
- `cnn_data/csv/calc_case_description_train_set.csv`
- `cnn_data/csv/calc_case_description_test_set.csv`
- imagens em `cnn_data/jpeg/` para executar o notebook de CNN

Os CSVs ficam versionados no repositório. As imagens de `cnn_data/jpeg/` não são versionadas por tamanho; mantenha essa pasta localmente quando quiser executar a CNN. Arquivos `*:Zone.Identifier` são metadados do Windows e são ignorados pelos notebooks.

## Saídas

Os notebooks salvam gráficos e resultados em `reports/`, incluindo distribuições, matrizes de confusão, importância de variáveis e curvas de treinamento.

## Observação clínica

Os modelos deste projeto são demonstrações acadêmicas de apoio à triagem. Eles não substituem avaliação médica, exames complementares ou decisão clínica profissional.
