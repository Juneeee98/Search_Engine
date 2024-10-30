import lucene
import time

from index import get_directory_size

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.search import TermQuery, IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.index import DirectoryReader, Term, IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, StringField, IntField
from org.apache.lucene.analysis.standard import StandardAnalyzer
from java.nio.file import Paths

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np

def cluster_businesses(searcher, tfidf_args, svd_args, norm_args, kmeans_args):
    tfidf_vectorizer = TfidfVectorizer(**tfidf_args)
    svd = TruncatedSVD(**svd_args)
    norm = Normalizer(**norm_args)
    kmeans = KMeans(**kmeans_args)
    
    reader = searcher.getIndexReader()
    n = reader.maxDoc()
    query = TermQuery(Term('doc_type', 'business'))    
    hits = searcher.search(query, n)
    print(f'Extracting Features From {len(hits)} Documents.')
    bids = []
    biz_features = []
    for hit in hits:
        doc = reader.document(hit.doc)
        categories = doc.get('categories') or ""
        name = doc.get('name')
        biz_features.append(name + " " + categories)
        bids.append(doc.get('business_id'))
    
    lsa = make_pipeline(tfidf_vectorizer, svd, norm)
    reduced = lsa.fit_transform(biz_features)
    print('Predicting Clusters...', end='')
    clusters = kmeans.fit_predict(reduced)
    print(' Complete.')
    return bids, clusters


def index_clusters(writer, searcher, bids, clusters):
    print('Indexing Clusters...', end='')
    reader = searcher.getIndexReader()
    for cluster, bid in zip(clusters, bids):
        bid_query = TermQuery(Term('business_id', bid))
        dt_query = TermQuery(Term('doc_type', 'business'))
        boolean_query = BooleanQuery.Builder()
        boolean_query.add(bid_query, BooleanClause.Occur.MUST)
        boolean_query.add(dt_query, BooleanClause.Occur.FILTER)
        query = boolean_query.build()
        hits = searcher.search(query, 1)
        hit = hits.scoreDocs[0]
        old_doc = reader.document(hit.doc)
        new_doc = Document()
        new_doc.add(StringField('business_id', bid, StringField.Store.YES))
        new_doc.add(StringField('cluster', str(int(cluster)), StringField.Store.YES))
        new_doc.add(StringField('doc_type', 'cluster', StringField.Store.YES))
        writer.addDocument(new_doc)
    print(' Complete.')


def create_secondary_index(index_directory, secondary_index_directory, tfidf_args, svd_args, norm_args, kmeans_args):
    searcher = IndexSearcher(DirectoryReader.open(FSDirectory.open(Paths.get(index_directory))))
    secondary_index = FSDirectory.open(Paths.get(secondary_index_directory))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
    writer = IndexWriter(secondary_index, config)

    bids, clusters = cluster_businesses(searcher, tfidf_args, svd_args, norm_args, kmeans_args)
    
    try:
        index_clusters(writer, searcher, bids, clusters)
        writer.commit()
    finally:
        writer.close()
  
    index_size = get_directory_size(secondary_index_directory)
    print(f"Index Size: {index_size / (1024 ** 2):.2f} MB")
            

if __name__ == "__main__":
    # Initialize the Lucene JVM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    # Define the path to the index directory
    index_directory = "./index"
    secondary_index_directory = "./secondary_index"
    
    tfidf_args = {
        'stop_words': 'english'
    }

    svd_args = {
        'n_components': 200, 
        'n_iter': 20,
        'random_state': 42
    }

    norm_args = {
        'copy': False    
    }

    kmeans_args = {
        'n_clusters': 50
    }

    # create_secondary_index
    create_secondary_index(index_directory, secondary_index_directory, tfidf_args, svd_args, norm_args, kmeans_args)
