# Databricks notebook source
# MAGIC %md このノートブックの目的は、Product Searchアクセラレータで使用するモデルを微調整することです。 このノートブックはhttps://github.com/databricks-industry-solutions/product-search

# COMMAND ----------

# MAGIC %md ##はじめに
# MAGIC
# MAGIC 一つ前のノートブックでは、セマンティック検索を実現するために埋め込みモデル(Embedding Model)とサンプルデータを使い、基本的な実装方法を示しました。
# MAGIC
# MAGIC 次は埋め込みモデルのファインチューニングを実行します。ファインチューニングにより、埋め込みモデルは特定のドメインに特化したデータセット（例えば、特定の製品カタログなど）に対して最適化されます。なお、モデルが事前学習で得た知識はそのまま残ります、加えて、提供された追加データから得た情報で補完されます。モデルが満足のいくまでチューニングされると、以前と同じようにパッケージ化され、永続化されます。
# MAGIC
# MAGIC ファインチューニングは、「教師あり」と「教師なし」の２つのアプローチがあります。本ノートブックでは、両方の実装方法を記載しています。

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install langchain-community==0.2.16 langchain-chroma==0.1.4 chromadb==0.4.24 "huggingface_hub<0.24"

# COMMAND ----------

# DBTITLE 1,Patch typing_extensions for DBR 14.3 compatibility
import typing_extensions
if not hasattr(typing_extensions, 'deprecated'):
  def _deprecated_shim(__msg, *, category=DeprecationWarning, stacklevel=1):
    def decorator(func):
      return func
    return decorator
  typing_extensions.deprecated = _deprecated_shim

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sentence_transformers import SentenceTransformer, util, InputExample, losses, evaluation
import torch
from torch.utils.data import DataLoader

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import numpy as np
import pandas as pd

import mlflow

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##ステップ1：ベースラインモデルのパフォーマンスを見積もる
# MAGIC
# MAGIC まずは、WANDSデータセットから、クエリと、それぞれのクエリで返された商品を取得します。 各クエリと商品の組み合わせについて、クエリと商品の整合性に基づいて割り当てられた数値スコアも取得します：

# COMMAND ----------

# DBTITLE 1,Get Search Results
# 検索に関連する製品テキストを組み立てる
search_pd = (
  spark   
    .table('products')
    .selectExpr(
      'product_id',
      'product_name',
      'COALESCE(product_description, product_name) as product_text' # use product description if available, otherwise name
      )
    .join(
      spark
        .table('labels'),
        on='product_id'
      )
    .join(
      spark
        .table('queries'),
        on='query_id'
      )
      .selectExpr('query','product_text','label_score as score')
  ).toPandas()

display(search_pd)

# COMMAND ----------

# MAGIC %md そして、前回のノートで使用したオリジナルモデルをダウンロードし、クエリと商品テキスト情報の両方を埋め込みに変換できるようにする：

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
# 埋め込みモデルのダウンロード
original_model = SentenceTransformer('all-MiniLM-L12-v2', device='cuda:0')

# COMMAND ----------

# DBTITLE 1,Convert Queries & Products to Embeddings
query_embeddings = (
  original_model
    .encode(
      search_pd['query'].tolist()
      )
  )

product_embeddings = (
  original_model
    .encode(
      search_pd['product_text'].tolist()
      )
  )

# COMMAND ----------

# MAGIC %md そして、クエリと製品間のコサイン類似度を計算します。埋め込み間の類似性は、2つのベクトル間の距離に関係するものとして語られますが、コサイン類似度は、空間の中心からベクトルによって識別される点（あたかも座標であるかのように）へ伸びる光線を区切る角度を指します。正規化されたベクトル空間では、この角度は点間の類似性の度合いも捉えます：

# COMMAND ----------

# DBTITLE 1,Calculate Cosine Similarity Between Queries and Products
# 各クエリと商品のペアについて、コサイン類似度を決定
original_cos_sim_scores = (
  util.pairwise_cos_sim(
    query_embeddings, 
    product_embeddings
    )
  )

# COMMAND ----------

# MAGIC %md これらの値を平均化することで、クエリが元の埋め込み空間のプロダクトにどれだけ近いかを知ることができます。 コサイン類似度の範囲は0.0から1.0であり、1.0に近づくにつれて値が向上することに注意してください：

# COMMAND ----------

# DBTITLE 1,Calculate Avg Cosine Similarity
# コサイン類似度の平均スコア
original_cos_sim_score = torch.mean(original_cos_sim_scores).item()

# 結果表示
print(original_cos_sim_score)

# COMMAND ----------

# MAGIC %md ラベルスコアとコサイン類似度の相関を調べることで、モデルの性能の別の尺度が得られます： 

# COMMAND ----------

# DBTITLE 1,Calculate Correlation with Scores 
# コサイン類似度と関連性スコアの相関を決定
original_corr_coef_score = (
  np.corrcoef(
    original_cos_sim_scores,
    search_pd['score'].values
  )[0][1]
) 
# 結果表示
print(original_corr_coef_score)

# COMMAND ----------

# MAGIC %md ##ステップ2：モデルのファインチューニング
# MAGIC
# MAGIC オリジナルモデルのパフォーマンスのベースラインのメトリックスが分かったので、次はアノテーションされた検索結果データを使ってモデルをファインチューニングすることができます。 まず、クエリ結果をモデルが必要とする入力リスト形式に再構築することから始めます：

# COMMAND ----------

# MAGIC %md ###ステップ2-A：教師あり学習編
# MAGIC
# MAGIC まずは教師あり学習のアプローチから：

# COMMAND ----------

# DBTITLE 1,Restructure Data for Model Input
# 入力を組み立てる関数を定義
def create_input(doc1, doc2, score):
  return InputExample(texts=[doc1, doc2], label=score)

# 各検索結果を入力に変換
inputs = search_pd.apply(
  lambda s: create_input(s['query'], s['product_text'], s['score']), axis=1
  ).to_list()

# COMMAND ----------

# MAGIC %md その後、チューニングするために、オリジナルモデルの別コピーをダウンロードします：

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
tuned_model = SentenceTransformer('all-MiniLM-L12-v2', device='cuda:0')

# COMMAND ----------

# MAGIC %md そして、コサイン類似度を最小化するようにモデルをチューニングします：
# MAGIC
# MAGIC **注意** このステップは、シングルノードクラスターに使用するサーバーをスケールアップすることで、より高速に実行されます。

# COMMAND ----------

# DBTITLE 1,Tune the Model
# モデルへの入力指示を定義
input_dataloader = DataLoader(inputs, shuffle=True, batch_size=16) # feed 16 records at a time to the model

# 最適化する損失指標を定義
loss = losses.CosineSimilarityLoss(tuned_model)

# 入力データに対してモデルをチューニング
tuned_model.fit(
  train_objectives=[(input_dataloader, loss)],
  epochs=1, # テストのためEpochは１に設定
  warmup_steps=100 # 学習率が最大まで上昇してからゼロに戻るまでのステップ数を制御
  )

# COMMAND ----------

# MAGIC %md モデルトレーニングの間、データに対して1パス（エポック）だけ実行するようにモデルを設定していることにお気づきでしょう。 この処理によって、実際にかなりの改善が見られますが、より多くの改善を求めるのであれば、この値を大きくして複数回のパスを行うこともできます。 *warmup_steps*の設定は、この領域でよく使われるものです。 他の値で実験するのも、デフォルトを使うのも自由です。

# COMMAND ----------

# MAGIC %md ###ステップ2-B：教師なし学習編
# MAGIC
# MAGIC 続いて教師なし学習のアプローチです：

# COMMAND ----------

# DBTITLE 1,Restructure Data for Model Input
# 入力を組み立てる関数を定義
def create_input_without_label(doc1, doc2):
  return InputExample(texts=[doc1, doc2])

# 各検索結果を入力に変換
inputs_without_label = search_pd.apply(
  lambda s: create_input_without_label(s['query'], s['product_text']), axis=1
  ).to_list()

# COMMAND ----------

# MAGIC %md その後、チューニングするために、オリジナルモデルの別コピーをダウンロードします：

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
tuned_model = SentenceTransformer('all-MiniLM-L12-v2', device='cuda:0')

# COMMAND ----------

# MAGIC %md そして、コサイン類似度を最小化するようにモデルをチューニングします：
# MAGIC
# MAGIC **注意** このステップは、シングルノードクラスターに使用するサーバーをスケールアップすることで、より高速に実行されます。

# COMMAND ----------

# DBTITLE 1,Tune the Model
# モデルへの入力指示を定義
input_without_label_dataloader = DataLoader(inputs_without_label, shuffle=True, batch_size=16) # feed 16 records at a time to the model

# 最適化する損失指標を定義
loss = losses.MultipleNegativesRankingLoss(tuned_model)

# 入力データに対してモデルをチューニング
tuned_model.fit(
  train_objectives=[(input_without_label_dataloader, loss)],
  epochs=1, # テストのためEpochは１に設定
  warmup_steps=100 # 学習率が最大まで上昇してからゼロに戻るまでのステップ数を制御
  )

# COMMAND ----------

# MAGIC %md モデルトレーニングの間、データに対して1パス（エポック）だけ実行するようにモデルを設定していることにお気づきでしょう。 実はこのデータセットでは教師なしでは１エポックだと期待通りの結果が得れないことがわかると思います。より多くの改善を求めるのであれば、この値を大きくして複数回のパスを行うことをお試しください。 *warmup_steps*の設定は、この領域でよく使われるものです。 他の値で実験するのも、デフォルトを使うのも自由です。

# COMMAND ----------

# MAGIC %md ##ステップ3：微調整されたモデルのパフォーマンスを見積もる
# MAGIC
# MAGIC モデルが調整されたので、前と同じようにそのパフォーマンスを評価することができます：

# COMMAND ----------

# DBTITLE 1,Calculate Cosine Similarities for Queries & Products in Tuned Model
query_embeddings = (
  tuned_model
    .encode(
      search_pd['query'].tolist()
      )
  )

product_embeddings = (
  tuned_model
    .encode(
      search_pd['product_text'].tolist()
      )
  )

# 各クエリと商品のペアについて、コサイン類似度を算出
tuned_cos_sim_scores = (
  util.pairwise_cos_sim(
    query_embeddings, 
    product_embeddings
    )
  )

# COMMAND ----------

# DBTITLE 1,Calculate Avg Cosine Similarity
# average the cosine similarity scores
tuned_cos_sim_score = torch.mean(tuned_cos_sim_scores).item()

# display result
print(f"With tuning, avg cosine similarity went from {original_cos_sim_score} to {tuned_cos_sim_score}")

# COMMAND ----------

# DBTITLE 1,Calculate Correlation Coefficient
# determine correlation between cosine similarities and relevancy scores
tuned_corr_coef_score = (
  np.corrcoef(
    tuned_cos_sim_scores,
    search_pd['score'].values
  )[0][1]
) 
# print results
print(f"With tuning, the correlation coefficient went from {original_corr_coef_score} to {tuned_corr_coef_score}")

# COMMAND ----------

# MAGIC %md これらの結果から、たった一度のデータ処理で、クエリーを製品に近づけ、モデルをデータの特殊性に合わせてチューニングしたことがわかります。 

# COMMAND ----------

# MAGIC %md ##ステップ4：デプロイのためにモデルを永続化する
# MAGIC
# MAGIC 前回と同じように、チューニングしたモデルをデータとともにパッケージ化し、永続化（そして最終的なデプロイ）を可能にします。 以下のステップは、前のノートブックと同じように、オリジナルのアセットとチューニングされたアセットを分離するために微調整を加えたものです：

# COMMAND ----------

# DBTITLE 1,Get Product Text to Search
# assemble product text relevant to search
product_text_pd = (
  spark
    .table('products')
    .selectExpr(
      'product_id',
      'product_name',
      'COALESCE(product_description, product_name) as product_text' # use product description if available, otherwise name
      )
  ).toPandas()

# COMMAND ----------

# DBTITLE 1,Load Product Info for Use with Encoder
# assemble product documents in required format (id, text)
documents = (
  DataFrameLoader(
    product_text_pd,
    page_content_column='product_text'
    )
    .load()
  )

# COMMAND ----------

# DBTITLE 1,Load Model as HuggingFaceEmbeddings Object
# encoder path (use local disk to avoid DBFS I/O issues)
import shutil, os
embedding_model_path = f"/local_disk0/tmp{config['dbfs_path']}/tuned_model"

# make sure path is clear
if os.path.exists(embedding_model_path):
  shutil.rmtree(embedding_model_path)
os.makedirs(embedding_model_path, exist_ok=True)

# reload model using langchain wrapper
tuned_model.save(embedding_model_path)
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

# COMMAND ----------

# DBTITLE 1,Generate Embeddings from Product Info
# chromadb path (use local disk to avoid DBFS SQLite I/O issues)
chromadb_path = f"/local_disk0/tmp{config['dbfs_path']}/tuned_chromadb"

# make sure chromadb path is clear
if os.path.exists(chromadb_path):
  shutil.rmtree(chromadb_path)
os.makedirs(chromadb_path, exist_ok=True)

# generate embeddings
vectordb = Chroma.from_documents(
  documents=documents, 
  embedding=embedding_model, 
  persist_directory=chromadb_path
  )

# chromadb 0.4.x persists automatically

# COMMAND ----------

# DBTITLE 1,Define Wrapper Class for Model
class ProductSearchWrapper(mlflow.pyfunc.PythonModel):


  # define steps to initialize model
  def load_context(self, context):

    # import required libraries
    import pandas as pd
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    # retrieve embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=context.artifacts['embedding_model'])

    # retrieve vectordb contents
    self._vectordb = Chroma(
      persist_directory=context.artifacts['chromadb'],
      embedding_function=embedding_model
      )

    # set number of results to return
    self._max_results = 5


  # define steps to generate results
  # note: query_df expects only one query
  def predict(self, context, query_df):


    # import required libraries
    import pandas as pd

    # perform search on embeddings
    raw_results = self._vectordb.similarity_search_with_score(
      query_df['query'].values[0], # only expecting one value at a time 
      k=self._max_results
      )

    # get lists of of scores, descriptions and ids from raw results
    scores, descriptions, names, ids = zip(
      *[(r[1], r[0].page_content, r[0].metadata['product_name'], r[0].metadata['product_id']) for r in raw_results]
      )

    # reorganized results as a pandas df, sorted on score
    results_pd = pd.DataFrame({
      'product_id':ids,
      'product_name':names,
      'product_description':descriptions,
      'score':scores
      }).sort_values(axis=0, by='score', ascending=True)
    
    # set return value
    return results_pd

# COMMAND ----------

# DBTITLE 1,Identify Model Artifacts
artifacts = {
  'embedding_model': embedding_model_path,
  'chromadb': chromadb_path
  }

print(
  artifacts
  )

# COMMAND ----------

# DBTITLE 1,Define Environment Requirements
import pandas

# get base environment configuration
conda_env = mlflow.pyfunc.get_default_conda_env()

# define packages required by model
packages = [
  f'pandas=={pandas.__version__}',
  'langchain-community==0.2.16',
  'langchain-chroma==0.1.4',
  'chromadb==0.4.24',
  'sentence-transformers'
  ]

# add required packages to environment configuration
conda_env['dependencies'][-1]['pip'] += packages

print(
  conda_env
  )

# COMMAND ----------

# DBTITLE 1,Persist Model
with mlflow.start_run() as run:

    mlflow.pyfunc.log_model(
        artifact_path='model', 
        python_model=ProductSearchWrapper(),
        conda_env=conda_env,
        artifacts=artifacts, # items at artifact path will be loaded into mlflow repository
        registered_model_name=config['tuned_model_name']
    )

# COMMAND ----------

# DBTITLE 1,Elevate to Production
client = mlflow.MlflowClient()

latest_version = client.get_latest_versions(config['tuned_model_name'], stages=['None'])[0].version

client.transition_model_version_stage(
    name=config['tuned_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# DBTITLE 1,Retrieve model from registry
model = mlflow.pyfunc.load_model(f"models:/{config['tuned_model_name']}/Production")

# COMMAND ----------

# DBTITLE 1,Test the Persisted Model
# construct search
search = pd.DataFrame({'query':['farmhouse dining room table']})

# call model
display(model.predict(search))

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
