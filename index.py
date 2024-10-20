import json
import lucene
import time
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, StringField, TextField, DoubleField, IntField
from org.apache.lucene.analysis.standard import StandardAnalyzer
from java.nio.file import Paths

# Initialize the Lucene JVM
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

def index_business_data(writer, business_data, business_map):
    """
    Indexes a single business data entry and populates business_map.
    """
    doc = Document()
    
    # Add basic fields with safe default values
    business_id = business_data.get("business_id", "")
    business_name = business_data.get("name", "Unknown Business")
    
    # Populate the business_map
    business_map[business_id] = business_name
    
    doc.add(StringField("business_id", business_id, StringField.Store.YES))
    doc.add(TextField("name", business_name, TextField.Store.YES))
    doc.add(TextField("address", business_data.get("address", ""), TextField.Store.YES))
    doc.add(TextField("city", business_data.get("city", ""), TextField.Store.YES))
    doc.add(StringField("state", business_data.get("state", ""), StringField.Store.YES))
    doc.add(StringField("postal_code", business_data.get("postal_code", ""), StringField.Store.YES))
    
    # Geospatial data
    doc.add(DoubleField("latitude", float(business_data.get("latitude", 0.0)), DoubleField.Store.YES))
    doc.add(DoubleField("longitude", float(business_data.get("longitude", 0.0)), DoubleField.Store.YES))
    
    # Additional business information
    doc.add(DoubleField("stars", float(business_data.get("stars", 0.0)), DoubleField.Store.YES))
    doc.add(IntField("review_count", int(business_data.get("review_count", 0)), IntField.Store.YES))
    
    # Check if categories field exists and is not None or empty
    categories = business_data.get("categories")
    if categories:
        doc.add(TextField("categories", categories, TextField.Store.YES))

    # Index the document
    writer.addDocument(doc)

def index_review_data(writer, review_data):
    """
    Indexes review data without filtering by business_id. The business ID mapping will be handled at search time.
    """
    doc = Document()

    # Link review to business by business_id
    business_id = review_data.get("business_id", "")
    doc.add(StringField("business_id", business_id, StringField.Store.YES))

    # Add the rest of the review fields
    doc.add(StringField("review_id", review_data["review_id"], StringField.Store.YES))
    doc.add(StringField("user_id", review_data["user_id"], StringField.Store.YES))
    doc.add(TextField("review_text", review_data.get("text", ""), TextField.Store.YES))
    doc.add(DoubleField("stars", float(review_data.get("stars", 0.0)), DoubleField.Store.YES))
    doc.add(IntField("useful", int(review_data.get("useful", 0)), IntField.Store.YES))
    doc.add(IntField("funny", int(review_data.get("funny", 0)), IntField.Store.YES))
    doc.add(IntField("cool", int(review_data.get("cool", 0)), IntField.Store.YES))

    # Add document to index
    writer.addDocument(doc)

def count_documents_in_file(file_path):
    """
    Counts the number of documents in a JSON file.
    """
    with open(file_path, "r") as file:
        return sum(1 for _ in file)

def load_and_index_json(writer, business_file_path, review_file_path, total_documents):
    """
    Loads business and review JSON data and indexes them, tracking time every 10% of documents.
    """
    # Index business data
    indexed_docs = 0
    start_time = time.time()
    interval = total_documents // 10  # Every 10%
    
    # Create a business_map to map business_id to business_name
    business_map = {}
    
    # Index businesses
    with open(business_file_path, "r") as business_file:
        for line in business_file:
            try:
                business_data = json.loads(line)
                index_business_data(writer, business_data, business_map)
                indexed_docs += 1

                # Log time every 10%
                if indexed_docs % interval == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Indexed {indexed_docs} / {total_documents} documents, Time elapsed: {elapsed_time:.2f} seconds")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"Error processing business entry: {e}")

    # Index reviews
    with open(review_file_path, "r") as review_file:
        for line in review_file:
            try:
                review_data = json.loads(line)
                index_review_data(writer, review_data)
                indexed_docs += 1

                # Log time every 10%
                if indexed_docs % interval == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Indexed {indexed_docs} / {total_documents} documents, Time elapsed: {elapsed_time:.2f} seconds")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"Error processing review entry: {e}")

def create_index(index_dir, business_json_file, review_json_file):
    """
    Create an index and index data from business and review JSON files.
    """
    index = FSDirectory.open(Paths.get(index_dir))

    # Set up the analyzer and the IndexWriter configuration
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(index, config)

    # Count total number of documents
    total_business_docs = count_documents_in_file(business_json_file)
    total_review_docs = count_documents_in_file(review_json_file)
    total_documents = total_business_docs + total_review_docs

    print(f"Total documents to index: {total_documents}")

    try:
        # Load and index the business and review data
        load_and_index_json(writer, business_json_file, review_json_file, total_documents)
        writer.commit()
    finally:
        writer.close()

if __name__ == "__main__":
    # Define the paths
    index_directory = "./index"  # Where your index will be stored
    business_json_file = "./dataset/yelp_academic_dataset_business.json"  # Path to business data
    review_json_file = "./dataset/yelp_academic_dataset_review.json"  # Path to review data

    # Create the index
    create_index(index_directory, business_json_file, review_json_file)