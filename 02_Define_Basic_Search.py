# Databricks notebook source
# MAGIC %md The purpose of this notebook is to transform product information for use in the Product Search accelerator.  You may find this notebook on https://github.com/databricks-industry-solutions/product-search.

# COMMAND ----------

# MAGIC %md ##はじめに
# MAGIC
# MAGIC データが揃ったので、次は既製のモデルを製品検索に適用する。この作業の重要な部分は、我々のモデルが推論中に製品カタログを高速に検索するために使用するベクトルデータベースの導入である。
# MAGIC
# MAGIC ベクトル・データベースを理解するためには、まず*埋め込み*を理解する必要がある。埋め込みとは、あるテキスト単位が、文書集合の中で頻繁に一緒に見つかる単語のクラスターとどの程度一致しているかを示す数値の配列である。これらの数値がどのように推定されるかについての正確な詳細は、ここではそれほど重要ではありません。 重要なのは、同じモデルによって生成された2つの埋め込み間の数学的距離が、2つの文書の類似性について何かを教えてくれることを理解することです。 私たちが検索を実行するとき、ユーザーの検索フレーズは埋め込みを生成するために使用され、それが私たちのカタログの製品に関連する既存の埋め込みと比較され、検索結果がどの埋め込みに最も近いかを決定します。それらの最も近いものが検索結果となる。
# MAGIC
# MAGIC 埋め込みの類似性を利用した商品の高速検索を実現するためには、埋め込みを格納するだけでなく、数値配列に対する高速検索を可能にする特殊なデータベースが必要です。このようなニーズに対応するデータストアのクラスはベクトルストアと呼ばれ、その中でも最も人気のあるものの1つが[Chroma](https://www.trychroma.com/)と呼ばれる軽量でファイルシステムベースのオープンソースストアです。 
# MAGIC
# MAGIC このノートブックでは、事前に訓練されたモデルをダウンロードし、このモデルを使用して商品テキストを埋め込みデータに変換し、埋め込みデータをChromaデータベースに格納します。

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
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import mlflow

import pandas as pd

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Assemble Product Info
# MAGIC
# MAGIC In this first step, we need to assemble the product text data against which we intend to search.  We will use our product description as that text unless there is no description in which case we will use the product name.  
# MAGIC
# MAGIC In addition to the searchable text, we will provide product metadata, such as product ids and names, that will be returned with our search results:
# MAGIC
# MAGIC
# MAGIC ステップ1：商品情報の組み立て
# MAGIC
# MAGIC この最初のステップでは、検索対象の商品テキストデータを作成する必要があります。 商品説明がない場合は商品名を使用します。 
# MAGIC
# MAGIC 検索可能なテキストに加え、検索結果で返される製品IDや製品名などの製品メタデータを提供します：

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

display(product_text_pd)

# COMMAND ----------

# MAGIC %md ##Step 2: Convert Product Info into Embeddings
# MAGIC
# MAGIC We will now convert our product text into embeddings.  The instructions for converting text into an embedding is captured in a language model.  The [*all-MiniLM-L12-v2* model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) is a *mini language model* (in contrast to a large language model) which has been trained on a large, well-rounded corpus of input text for good, balanced performance in a variety of document search scenarios.  The benefit of the *mini* language model as compared to a *large* language is that the *mini* model generates a more succinct embedding structure that facilitates faster search and lower overall resource utilization.  Given the limited breadth of the content in a product catalog, this is the best option of our needs:
# MAGIC
# MAGIC
# MAGIC ステップ2：商品情報を埋め込みに変換する
# MAGIC
# MAGIC これから、商品テキストをエンベッディングに変換します。 テキストを埋め込みに変換する命令は、言語モデルに取り込まれます。 [*all-MiniLM-L12-v2*モデル](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)は、(大規模な言語モデルとは対照的に)*ミニ言語モデル*であり、様々な文書検索シナリオにおいてバランスの取れた優れた性能を発揮するために、大規模で充実した入力テキストコーパスで学習されています。 大きな言語モデルと比較した場合の*ミニ*言語モデルの利点は、*ミニ*モデルがより簡潔な埋め込み構造を生成することで、より高速な検索と全体的なリソース使用率の低下を促進することである。 製品カタログのコンテンツの幅が限られていることを考えると、これは私たちのニーズに最適な選択肢です：

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
# download embeddings model
original_model = SentenceTransformer('all-MiniLM-L12-v2')

# COMMAND ----------

# MAGIC %md To use our model with our vector store, we need to wrap it as a LangChain HuggingFaceEmbeddings object.  We could have had that object download the model for us, skipping the previous step, but if we had done that, future references to the model would trigger additional downloads.  By downloading it, saving it to a local path, and then having the LangChain object read it from that path, we are bypassing unnecessary future downloads:
# MAGIC
# MAGIC
# MAGIC モデルをベクターストアで使うには、モデルをLangChainのHuggingFaceEmbeddingsオブジェクトとしてラップする必要があります。 このオブジェクトにモデルをダウンロードさせ、前のステップをスキップすることもできますが、そうすると、今後モデルを参照したときに、さらにダウンロードが必要になります。 ダウンロードし、ローカルパスに保存し、LangChainオブジェクトにそのパスから読み込ませることで、不必要な将来のダウンロードを回避しています：

# COMMAND ----------

# DBTITLE 1,Load Model as HuggingFaceEmbeddings Object
# encoder path (use local disk to avoid DBFS I/O issues)
embedding_model_path = f"/local_disk0/tmp{config['dbfs_path']}/embedding_model"

# make sure path is clear
if os.path.exists(embedding_model_path):
  shutil.rmtree(embedding_model_path)
os.makedirs(embedding_model_path, exist_ok=True)

# reload model using langchain wrapper
original_model.save(embedding_model_path)
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

# COMMAND ----------

# MAGIC %md Using our newly downloaded model, we can now generate embeddings.  We'll persist these to the Chroma vector database, a database that will allow us to retrieve vector data efficiently in future steps:
# MAGIC
# MAGIC 新しくダウンロードしたモデルを使って、埋め込みデータを生成します。 これをChromaベクトル・データベースに永続化し、今後のステップでベクトル・データを効率的に取得できるようにする：

# COMMAND ----------

# DBTITLE 1,Reset Chroma File Store
# chromadb path (use local disk to avoid DBFS SQLite I/O issues)
chromadb_path = f"/local_disk0/tmp{config['dbfs_path']}/chromadb"

# make sure chromadb path is clear
import shutil, os
if os.path.exists(chromadb_path):
  shutil.rmtree(chromadb_path)
os.makedirs(chromadb_path, exist_ok=True)

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

# DBTITLE 1,Generate Embeddings from Product Info
# define logic for embeddings storage
vectordb = Chroma.from_documents(
  documents=documents, 
  embedding=embedding_model, 
  persist_directory=chromadb_path
  )

# chromadb 0.4.x persists automatically

# COMMAND ----------

# MAGIC %md From a count of the vector database collection, we can see that every product entry from our original dataframe has been loaded:
# MAGIC
# MAGIC ベクトルデータベースコレクションのカウントから、元のデータフレームからすべての製品エントリがロードされていることがわかります：

# COMMAND ----------

# DBTITLE 1,Count Items in Vector DB
vectordb._collection.count()

# COMMAND ----------

# MAGIC %md We can also take a peek at one of the records in the database to see how our data has been transformed.  Details about our product id and product name, basically all the fields in the original dataframe not identified as the *document* are stored in the *Metadatas* field.  The text we identified as our *document* is visible in its original form through the *Documents* field and the embedding created from this is available through the *embeddings* field:
# MAGIC
# MAGIC
# MAGIC データベースのレコードの1つを覗いて、データがどのように変換されたかを確認することもできます。 製品 ID と製品名に関する詳細、基本的に元のデータフレームの *ドキュメント* として識別されていないすべてのフィールドは、*Metadatas* フィールドに格納されています。 私たちが *document* として識別したテキストは、*Documents* フィールドを通して元の形で見ることができ、ここから作成された埋め込みは、*embeddings* フィールドを通して見ることができます：

# COMMAND ----------

# DBTITLE 1,Examine a Vector DB record
rec= vectordb._collection.peek(1)

print('Metadatas:  ', rec['metadatas'])
print('Documents:  ', rec['documents'])
print('ids:        ', rec['ids'])
print('embeddings: ', rec['embeddings'])

# COMMAND ----------

# MAGIC %md ##Step 3: Demonstrate Basic Search Capability
# MAGIC
# MAGIC To get a sense of how our search will work, we can perform a similarity search on our vector database:
# MAGIC
# MAGIC ステップ3：基本的な検索機能のデモンストレーション
# MAGIC
# MAGIC 検索がどのように機能するかを知るために、ベクターデータベースの類似性検索を行ってみましょう：

# COMMAND ----------

# DBTITLE 1,Perform Simple Search
vectordb.similarity_search_with_score("kid-proof rug")

# COMMAND ----------

# MAGIC %md Notice that while some of the results reflect key terms, such as *kid*, some do not.  This form of search is leveraging embeddings which understand that terms like *child*, *children*, *kid* and *kids* often are associated with each other. And while the exact term *kid* doesn't appear in every result, the presence of *children* indicates that at least one of the results is close in concept to what we are searching for.
# MAGIC
# MAGIC 結果の中には、*kid*のようなキーワードが反映されているものもあれば、反映されていないものもある。 この検索形式は、*child*、*children*、*kid*、*kids*のような用語がしばしば互いに関連していることを理解する埋め込みを活用している。そして、*kid*という正確な用語がすべての検索結果に現れるわけではないが、*children*が存在するということは、検索結果の少なくともひとつが、私たちが探しているものに近い概念であることを示している。

# COMMAND ----------

# MAGIC %md ##Step 4: Persist Model for Deployment
# MAGIC
# MAGIC At this point, we have all the elements in place to build a deployable model.  In the Databricks environment, deployment typically takes place using [MLflow](https://www.databricks.com/product/managed-mlflow), which has the ability to build a containerized service from our model as one of its deployment patterns.  Generic Python models deployed with MLflow typically support a standard API with a *predict* method that's called for inference.  We will need to write a custom wrapper to map a standard interface to our model as follows:
# MAGIC
# MAGIC
# MAGIC ステップ4：デプロイメントのためにモデルを永続化する
# MAGIC
# MAGIC この時点で、デプロイ可能なモデルを構築するための全ての要素が揃いました。 Databricks環境では、デプロイは通常[MLflow](https://www.databricks.com/product/managed-mlflow)を使って行われます。MLflowはデプロイパターンの一つとして、モデルからコンテナ化されたサービスを構築する機能を持っています。 MLflowでデプロイされる一般的なPythonモデルは、推論のために呼び出される*predict*メソッドを持つ標準的なAPIをサポートしています。 以下のように、標準インターフェイスをモデルにマッピングするためのカスタムラッパーを書く必要があります：

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

# MAGIC %md The *load_context* of the previously defined wrapper class addresses the steps that need to take place at model initialization. Two of those steps make reference to artifacts within the model's context.  
# MAGIC
# MAGIC Artifacts are assets stored with the model as it is logged with MLflow.  Using keys assigned to these artifacts, those assets can be retrieved for utilization at various points in the model's logic. 
# MAGIC
# MAGIC The two artifacts needed for our model are the path to the saved model and the Chroma database, both of which were persisted to storage in previous steps.  Please note that these objects were saved to the *Databricks Filesystem* which MLflow understands how to reference.  As a result, we need to alter the paths to these items by replacing the local */dbfs* to *dbfs:*: 
# MAGIC
# MAGIC
# MAGIC
# MAGIC 先に定義したラッパークラスの *load_context*は、モデルの初期化時に必要なステップに対応しています。これらのステップのうち2つは、モデルのコンテキスト内でアーティファクトを参照します。 
# MAGIC
# MAGIC アーティファクト（成果物）とは、MLflowでログに記録されるモデルとともに保存される資産のことです。 これらのアーティファクトに割り当てられたキーを使用することで、モデルのロジックの様々なポイントで利用するために、これらのアセットを取り出すことができます。
# MAGIC
# MAGIC このモデルに必要な2つのアーティファクトは、保存されたモデルへのパスとChromaデータベースです。 これらのオブジェクトは、MLflowが参照方法を理解している*Databricks Filesystem*に保存されていることに注意してください。 その結果、ローカルの*/dbfs*を*dbfs:*に置き換えることで、これらのアイテムへのパスを変更する必要があります： 

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

# MAGIC %md Next, we need to identify the packages we need to insure are installed as our model is deployed.  These are:
# MAGIC
# MAGIC 次に、モデルのデプロイ時にインストールされることを保証する必要があるパッケージを特定する必要がある。 これらは以下の通りです：

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

# MAGIC %md Now we can persist our model to MLflow.  Notice that in this scenario, our embedding model and Chroma database are being loaded as artifacts and that our *python_model* is just the class definition that provides the logic for hydrating a model from those artifacts:
# MAGIC
# MAGIC これでモデルをMLflowに永続化できます。 このシナリオでは、エンベッディングモデルとChromaデータベースがアーティファクトとしてロードされ、*python_model*はこれらのアーティファクトからモデルをハイドレートするためのロジックを提供するクラス定義に過ぎないことに注意してください：

# COMMAND ----------

# DBTITLE 1,Persist Model to MLflow
with mlflow.start_run() as run:

    mlflow.pyfunc.log_model(
        artifact_path='model',
        python_model=ProductSearchWrapper(),
        conda_env=conda_env,
        artifacts=artifacts, # items at artifact path will be loaded into mlflow repository
        registered_model_name=config['basic_model_name']
    )

# COMMAND ----------

# MAGIC %md If we use the experiments UI (accessible by clicking the flask icon in the right-hand navigation of your workspace), we can access the details surrounding the model we just logged.  By expanding the folder structure behind the model, we can see the model and vector store assets loaded into MLflow:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_mlflow_artifacts.PNG'>
# MAGIC
# MAGIC
# MAGIC 実験UI（ワークスペースの右側のナビゲーションにあるフラスコアイコンをクリックすることでアクセス可能）を使用すると、先ほどログに記録したモデルの詳細にアクセスすることができます。 モデルの背後にあるフォルダ構造を展開すると、MLflowにロードされたモデルとベクターストアのアセットを見ることができます：
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_mlflow_artifacts.PNG'>

# COMMAND ----------

# MAGIC %md We can now elevate our model to production status.  This would typically be done through a careful process of testing and evaluation but for the purposes of this demo, we'll just programmatically push it forward:
# MAGIC
# MAGIC
# MAGIC これでモデルをプロダクション・ステータスに昇格させることができる。 これは通常、テストと評価の慎重なプロセスを経て行われるものですが、このデモの目的では、プログラム的に進めるだけです：

# COMMAND ----------

# DBTITLE 1,Elevate to Production
client = mlflow.MlflowClient()

latest_version = client.get_latest_versions(config['basic_model_name'], stages=['None'])[0].version

client.transition_model_version_stage(
    name=config['basic_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md Loading our model, we can perform a simple test to see results from a sample search.  
# MAGIC
# MAGIC モデルをロードして、サンプル検索の結果を見るための簡単なテストを実行できる。 

# COMMAND ----------

# DBTITLE 1,Retrieve model from registry
model = mlflow.pyfunc.load_model(f"models:/{config['basic_model_name']}/Production")

# COMMAND ----------

# MAGIC %md If you are curious why we are constructing a pandas dataframe for our search term, please understand that this aligns with how data will eventually passed to our model when we host it in model serving.  The logic in our *predict* function anticipates this as well.
# MAGIC
# MAGIC Inferencing a single record can take approximately 50-300 ms, allowing the model to be served and used by a user-facing webapp. 
# MAGIC
# MAGIC
# MAGIC なぜ検索語のためにpandasのデータフレームを構築しているのか気になる方は、モデルサービングでデータをホストする際に、データが最終的にどのようにモデルに渡されるかに合わせていることをご理解ください。 *predict*関数のロジックもこれを想定しています。
# MAGIC
# MAGIC 1つのレコードを推論するのにかかる時間は約50-300ミリ秒であり、ユーザー向けのウェブアプリケーションでモデルを提供し、使用することができます。

# COMMAND ----------

# DBTITLE 1,Test Persisted Model with Sample Search
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
