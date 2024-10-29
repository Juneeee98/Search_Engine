import json
import lucene
import os
import time
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, StringField, TextField, DoubleField, IntField, Field
from org.apache.lucene.analysis.standard import StandardAnalyzer
from java.nio.file import Paths


def index_business_data(writer, business_data, business_map):
    """
    Indexes a single business data entry and populates business_map.
    """
    doc = Document()
    
    # Add basic fields with safe default values
    business_id = business_data.get("business_id", "")
    business_name = business_data.get("name", "Unknown Business")
    business_categories = business_data.get("categories") or ""
    
    # Populate the business_map
    business_map[business_id] = business_name
    
    doc.add(StringField('doc_type', 'business', StringField.Store.YES))
    doc.add(StringField("business_id", business_id, StringField.Store.YES))
    doc.add(TextField("name", business_name, TextField.Store.YES))
    doc.add(TextField("categories", business_categories, TextField.Store.YES))
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
    

    # Index the document
    writer.addDocument(doc)

def index_review_data(writer, review_data):
    """
    Indexes review data without filtering by business_id. The business ID mapping will be handled at search time.
    """
    doc = Document()

    # Link review to business by business_id
    business_id = review_data.get("business_id", "")
    doc.add(StringField('doc_type', 'review', StringField.Store.YES))
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

def get_file_size(filepath):
    """
    Returns the size of a file in bytes.
    """
    return os.path.getsize(filepath)

def get_directory_size(directory):
    """
    Returns the size of a directory in bytes.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

def load_and_index_json(writer, business_file_path, review_file_path, total_documents):
    """
    Loads business and review JSON data and indexes them, tracking time every 10% of documents.
    """
    # Index business data
    indexed_docs = 0
    start_time = time.time()
    last_segment_time = start_time
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
                    curr_time = time.time()
                    segment_elapsed_time = curr_time - last_segment_time
                    total_elapsed_time = curr_time - start_time
                    print(f"Indexed {indexed_docs} / {total_documents} documents, Segment time: {segment_elapsed_time:.2f} seconds, Total time elapsed: {total_elapsed_time:.2f} seconds")
                    last_segment_time = curr_time
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
                    curr_time = time.time()
                    segment_elapsed_time = curr_time - last_segment_time
                    total_elapsed_time = curr_time - start_time
                    print(f"Indexed {indexed_docs} / {total_documents} documents, Segment time: {segment_elapsed_time:.2f} seconds, Total time elapsed: {total_elapsed_time:.2f} seconds")
                    last_segment_time = curr_time
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

    # Calculate original data size
    business_data_size = get_file_size(business_json_file)
    review_data_size = get_file_size(review_json_file)
    original_data_size = business_data_size + review_data_size

    # Calculate index size
    index_size = get_directory_size(index_dir)

    # Display size comparison
    print(f"\nOriginal Data Size: {original_data_size / (1024 ** 2):.2f} MB")
    print(f"Index Size: {index_size / (1024 ** 2):.2f} MB")
    print(f"Reduction Factor: {(original_data_size / index_size):.2f}")

def filter_dataset_by_state(json_directory="./dataset", state="ID"):
    """
    Filters the business dataset by state and saves the filtered data to a new file.
    """
    # Load the business data
    business_file_path = os.path.join(json_directory, "yelp_academic_dataset_business.json")
    filtered_file_path = os.path.join(json_directory, f"{state.lower()}_business.json")

    # Filter the business data by state
    print(f"Filtering business data by state: {state}")
    business_ids = set()
    with open(business_file_path, "r") as business_file, open(filtered_file_path, "w") as filtered_file:
        for line in business_file:
            business_data = json.loads(line)
            if business_data.get("state") == state:
                filtered_file.write(json.dumps(business_data) + "\n")
                business_ids.add(business_data.get("business_id"))

    # Load reviews data
    review_file_path = os.path.join(json_directory, "yelp_academic_dataset_review.json")
    filtered_review_file_path = os.path.join(json_directory, f"{state.lower()}_review.json")

    # Filter the review data by state
    print(f"Filtering review data by state: {state}")
    with open(review_file_path, "r") as review_file, open(filtered_review_file_path, "w") as filtered_review_file:
        for line in review_file:
            review_data = json.loads(line)
            if review_data.get("business_id") in business_ids:
                filtered_review_file.write(json.dumps(review_data) + "\n")

    print(f"Filtered business data saved to {filtered_file_path}")
    print(f"Filtered review data saved to {filtered_review_file_path}")

if __name__ == "__main__":
    # Initialize the Lucene JVM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    # Define the paths
    index_directory = "./index"  # Where your index will be stored
    business_json_file = "./dataset/yelp_academic_dataset_business.json"  # Path to business data
    review_json_file = "./dataset/yelp_academic_dataset_review.json"  # Path to review data

    # Check if the dataset directory exists
    assert os.path.exists("./dataset"), f"Dataset directory not found at ./dataset"

    # Filter the dataset by state
    if not os.path.exists("./dataset/id_business.json"):
        filter_dataset_by_state(json_directory="./dataset", state="ID")
    # Reassign files to
    business_json_file = "./dataset/id_business.json"
    review_json_file = "./dataset/id_review.json"

    # Create the index and display size comparison
    create_index(index_directory, business_json_file, review_json_file)
