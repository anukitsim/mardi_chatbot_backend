from docx import Document
import json
import os

def extract_faqs(doc_path):
    """
    Extracts question-answer pairs from a given .docx FAQ document.
    """
    document = Document(doc_path)
    qas = []
    question = None
    answer = ""  # Initialize answer to handle the first question properly

    # Read through each paragraph in the document
    for para in document.paragraphs:
        text = para.text.strip()
        
        # Check if the line ends with a "?" indicating a question
        if text.endswith("?"):
            if question:
                # Save the previous question-answer pair
                qas.append({"question": question, "answer": answer.strip()})
            # Start a new question and reset the answer
            question = text
            answer = ""
        else:
            # Accumulate text lines for the answer
            answer += " " + text

    # Append the last question-answer pair if it exists
    if question:
        qas.append({"question": question, "answer": answer.strip()})
    
    return qas

def process_documents(input_folder, output_folder):
    """
    Processes all .docx files in the input folder and saves extracted Q&A as JSON files in the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".docx"):
            doc_path = os.path.join(input_folder, filename)
            qas = extract_faqs(doc_path)
            
            # Save the extracted Q&A pairs as a JSON file
            json_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_folder, json_filename)
            
            with open(output_path, "w") as f:
                json.dump(qas, f, indent=2)
            
            print(f"Q&A pairs extracted and saved to {output_path}")

# Run the script
if __name__ == "__main__":
    # Define input and output folders
    input_folder = "uploaded_documents"
    output_folder = "processed_json"
    
    # Process all documents in the input folder
    process_documents(input_folder, output_folder)
