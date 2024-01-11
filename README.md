# Boas Práticas no Desenvolvimento de Projetos em Ciência de Dados

Este projeto é dedicado ao reforço das boas práticas no desenvolvimento de pipelines de Machine Learning (ML). O repositório inclui código para treinamento e inferência com um modelo de aprendizado profundo para classificação de imagens. O pipeline se concentra em dados de imagens e demonstra práticas-chave para treinamento de modelos, avaliação e inferência.

### Sumário
- [Boas Práticas no Desenvolvimento de Projetos em Ciência de Dados](#boas-práticas-no-desenvolvimento-de-projetos-em-ciência-de-dados)
    - [Sumário](#sumário)
  - [Estrutura de Diretórios](#estrutura-de-diretórios)
  - [Pipeline de Treinamento](#pipeline-de-treinamento)
  - [Pipeline de Inferência](#pipeline-de-inferência)
  - [Requisitos](#requisitos)
  - [Uso](#uso)
  - [Contribuições](#contribuições)
  - [Licença](#licença)

## Estrutura de Diretórios

O diretório do projeto está organizado da seguinte forma:

- architectures/img/classification: Contém a arquitetura de rede neural para classificação de imagens (DefaultImageClassificationNet).
- img/classification: Inclui funções de utilidade para manipulação de dados de imagens (compile_img_data e ImageClassificationDataset).
-utils: Armazena funções utilitárias comuns e constantes (constants.py e model.py).
- notebooks: Espaço para notebooks Jupyter relacionados ao projeto.
- data: Dados de exemplo para treinamento e inferência.
- models: Diretório para salvar modelos treinados.
- reference_classes.csv: Arquivo CSV contendo as classes de referência para inferência.

## Pipeline de Treinamento

O pipeline de treinamento é implementado no arquivo train.py. Ele segue as melhores práticas para o desenvolvimento de ML:

- Modularização: O código é organizado em funções utilitárias e classes.
- Configurabilidade: Hiperparâmetros são configuráveis por meio de argumentos da linha de comando usando argparse.
- Manipulação de Dados: Os dados de imagem são compilados e transformados usando funções utilitárias.
- Arquitetura do Modelo: É utilizado um modelo padrão de rede neural para classificação de imagens (DefaultImageClassificationNet).
- Carregamento de Dados: DataLoader é empregado para o carregamento eficiente de dados durante o treinamento.
- Treinamento do Modelo: O modelo é treinado usando o otimizador Adam e a função de perda CrossEntropyLoss.
- Registro e Monitoramento: O progresso do treinamento é exibido com uma barra de progresso.

## Pipeline de Inferência

O pipeline de inferência é implementado no arquivo predict.py. Práticas-chave incluem:

- Configurabilidade: Parâmetros de inferência são configuráveis por meio de argumentos da linha de comando.
- Carregamento do Modelo: O modelo treinado é carregado para inferência usando a função load_model.
- Carregamento de Dados: DataLoader é empregado para o carregamento eficiente de dados durante a inferência.
- Predição e Exportação: Resultados da inferência, incluindo predições e probabilidades, são exportados para um arquivo CSV.

## Requisitos

Certifique-se de ter as dependências necessárias instaladas. Você pode instalá-las usando o seguinte comando:

```bash
conda env create -f env/environment.yml
conda activate treinamento-ml
```

## Uso
<ol>
<li><h3>Treinamento:</h3></li>

```bash
python train.py --epochs 10 --batch_size 32 --learning_rate 0.001 --path ../data/cat_vs_dog --save_path ../models/model.pth
```
- `--epochs`: Número de épocas de treinamento.
- `--batch_size`: Tamanho do lote para treinamento.
- `--learning_rate`: Taxa de aprendizado para treinamento.
- `--path`: Caminho para dados contendo pastas "train" e "val" (opcional).
- `--save_path`: Caminho para salvar o modelo treinado.

<li><h3>Inferência</h3></li>

```bash
python predict.py --batch_size 32 --path ../data/cat_vs_dog/prediction --model_path ../models/model.pth --reference_classes ./reference_classes.csv --save_path ./
```

- `--batch_size`: Tamanho do lote para inferência.
- `--path`: Caminho para a pasta "prediction".
- `--model_path`: Caminho para o modelo treinado.
- `--reference_classes`: Caminho para o CSV de classes de referência.
- `--save_path`: Caminho onde o arquivo CSV com os resultados da inferência será salvo.


## Contribuições

Contribuições são bem-vindas! Se você tiver sugestões, relatórios de bugs ou novos recursos, crie uma issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo [LICENSE](LICENSE) para obter detalhes.
