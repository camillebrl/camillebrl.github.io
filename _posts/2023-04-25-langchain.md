---
layout: post
title: Langchain
date: 2023-04-25
---

**[Langchain](https://python.langchain.com/en/latest/index.html) est un outil qui simplifie l'utilisation des LLMs dans différents use-cases. Cet outil chaîne différents composants des LLMs pour créer des uses-cases plus avancés qui utilisent ces LLMs.**



## Langchain permet:
- Une gestion des prompts
- De créer des sequences d’appels aux LLMs, via n'importe quel LLM provider ([HuggingFace Hub](https://huggingface.co/models), [OpenAI](https://platform.openai.com/docs/models/overview), [Cohere](https://docs.cohere.com/docs/the-cohere-platform), [AI21](https://www.ai21.com/studio/foundation-models), [Azure OpenAI](https://azure.microsoft.com/fr-fr/products/cognitive-services/openai-service), [Goose AI](https://goose.ai/docs), ..., mais aussi des modèles stockés localement). Pour voir comment utiliser des modèles de chaque provider, voir la [documentation](https://python.langchain.com/en/latest/ecosystem.html).
- D'avoir une interface standard pour les agents qui utilisent les LLMs pour faire des choses
- De charger des document qui seront utilisés par les LLMs simplement : tous les types (File loader & directory loader), Notion, Youtube, PowerPoint, HTML, Notebooks, … Ces documents sont ensuite traités à l'aide d'[UnstructuredIO](https://github.com/Unstructured-IO/unstructured)

## Utilisation des index pour faire des requêtes sur des documents:
{% highlight lang %}
from langchain.document_loader import TextLoader
loader = TextLoader("state_of_the_union.txt")
{% endhighlight %}
{% highlight lang %}
from langchain.indexes import VectorStorIndexCreator
index = VectorStoreIndexCreator().from_loader([loader])
{% endhighlight %}
Ici, il est possible d'utiliser plusieurs [Vectorstores](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html): [ElasticSearch](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/elasticsearch.html), [Qdrant](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/qdrant.html), [Redis](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/redis.html), [Chroma](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html), [AtlasDB](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/atlas.html), ...
{% highlight lang %}
query = "What did the president say about Ketanji Brown Jackson"
index.query(query)
{% endhighlight %}
{% highlight lang %}
Output: "The president said that Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in the private practive, a former federal public defender, and from a family of public school educators and pilice officers. He also said that she is a consensus builder and has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans."
{% endhighlight %}

Il est possible de choisir le modèle utilisé pour requêter sur e document. Voici un exemple utilisant le modèle text-davinci-003 ansi que le système de vectorindex [GPTSimpleVectorIndex](https://gpt-index.readthedocs.io/en/latest/guides/primer/usage_pattern.html) de [LlamaIndex](https://github.com/jerryjliu/llama_index?ref=alphasec.io). Dans cet exemple, l'outil est déployé sur Streamlit [Streamlit](https://streamlit.io/) de manière très simple:
{% highlight lang %}
import os, streamlit as st
# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= ""
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
from langchain import OpenAI
# This example uses text-davinci-003 by default; feel free to change if desired
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
# Configure prompt parameters and initialise helper
max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
# Load documents from the 'data' directory
documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)
# Define a simple Streamlit app
st.title("Ask Llama")
query = st.text_input("What would you like to ask?", "")
if st.button("Submit"):
    response = index.query(query)
    st.write(response)
{% endhighlight %}

## Utilisation des Retrievers pour faire des requêtes sur plusieurs documents :
Un [Retriever](https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/chatgpt-plugin-retriever.html 
) est le composant qui utilise l’index pour trouver en renvoyer des documents pertinents en réponse à la requête d’un utilisateur. Par exemple, on a une base de données contenant plusieurs documents, on ne connait pas le document dans lequel se trouve une réponse. A l'aide des Retrievers et de langchain, on peut simplement déployer un chatbot qui se base sur une base de connaissance.
On va prendre l'exemple du [ChatGPT Retrieval Plugin](https://github.com/openai/chatgpt-retrieval-plugin). Le plugin de récupération cherche à répondre aux questions ou aux besoins des utilisateurs en recherchant des documents pertinents dans une base de données vectorielle. Il fonctionne comme ceci:
- Ajoute les documents à une base de données vectorielle qui stocke les textes des documents, ainsi que l'embedding correspondant de ces textes qui est construit avec text-embedding-ada-002
- Découpage par morceaux de 200 tokens des bouts de texte
- Comparaison (distance cosinus) des morceaux de 200 tokens avec l'embedding de la question / de l'input de l'utilisateur pour trouver le bout de texte contenant la réponse
- Application du QA sur ce bout de texte

Il est possible de choisir sa base de données vectorielle avec le ChatGPT Retrieval Plugin: [ElasticSearch](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/elasticsearch.html), [Qdrant](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/qdrant.html), [Redis](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/redis.html), [Chroma](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html), [AtlasDB](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/atlas.html), ...

![Exemple du ChatGPT Plugin Retriever](/assets/images/chatgptpluginretriever.png)

