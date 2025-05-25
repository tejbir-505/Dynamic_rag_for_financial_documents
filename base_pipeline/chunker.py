import re
import os
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

from pdf_extracter import process_pdf 

import nltk
# nltk.data.path  
nltk.download('punkt_tab', download_dir='punkt_tab')
nltk.data.path.append('punkt_tab')
from nltk.tokenize import sent_tokenize

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemanticChunk:
    """Data class for storing semantic chunks with metadata"""
    content: str
    section_title: str
    page_numbers: List[int]
    start_sentence: int
    end_sentence: int
    chunk_id: str
    contextual_header: str
    original_content: str  # Content without contextual header

class SemanticChunker:
    """Semantic chunker with AutoContext for RAG pipeline"""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", max_chunk_size: int = 1000):
        """
        Initialize the semantic chunker
        
        Args:
            model_name: Groq model name for LLM
            max_chunk_size: Maximum size for final chunks in characters
        """
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model_name,
            temperature=0.1
        )
        
        self.max_chunk_size = max_chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def chunk_and_enumerate_sentences(self, pdf_result: Dict[str, Any]) -> Tuple[str, List[Dict]]:
        """
        Takes the result from process_pdf, and returns sentence-enumerated text.
        Format: <1> Sentence one. <2> Sentence two.
        """
        combined_text = pdf_result["combined_text"]
        pages_map = pdf_result["pages_map"]
        
        if not combined_text:
            logger.warning("No combined_text found in PDF result.")
            return "", pages_map

        sentences = sent_tokenize(combined_text)
        enumerated_text = ""
        for i, sentence in enumerate(sentences, 1):
            enumerated_text += f"<{i}> {sentence.strip()}\n"
        return enumerated_text, pages_map
    
    def get_semantic_sections(self, enumerated_text: str, document_title: str = "") -> List[Dict]:
        """
        Use LLM to identify semantic sections in the enumerated text
        
        Args:
            enumerated_text: Text with sentence numbers
            document_title: Title of the document for context
            
        Returns:
            List of section dictionaries with start_line, end_line, and title
        """
        
        system_prompt = """You are an expert document analyzer. Your task is to identify semantically cohesive sections in a document.

Instructions:
1. Analyze the provided text where each sentence is annotated with a unique number in the format <1>, <2>, etc.
2. The text may include table data represented in plain text (i.e., without clear rows/columns). Identify and treat such content carefully while determining cohesive topics.
3. Identify groups of sentences that form semantically cohesive sections (i.e., related topics, concepts, or themes).
4. Each section should be substantial, ideally covering a few paragraphs to a few pages worth of content.
5. For each section, provide:
   - The starting sentence number
   - The ending sentence number
   - A short but descriptive title that reflects the section’s main theme
6. Ensure the entire document is covered without overlaps between sections.

Return your output strictly in the following JSON format:
[
    {
        "start_sentence": 1,
        "end_sentence": 15,
        "title": "Descriptive Section Title"
    },
    {
        "start_sentence": 16,
        "end_sentence": 30,
        "title": "Another Descriptive Section Title"
    }
]

Only return the JSON array, no additional explanation or commentary."""

        user_prompt = f"Document Title: {document_title}\n\nEnumerated Text:\n{enumerated_text}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            # Extract JSON from response
            content = response.content.strip()
            
            # Try to find JSON in the response
            if content.startswith('['):
                sections = json.loads(content)
            else:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    sections = json.loads(json_match.group(1))
                else:
                    # Try to find JSON array in the text
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        sections = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No valid JSON found in LLM response")
            
            logger.info(f"Identified {len(sections)} semantic sections")
            return sections
            
        except Exception as e:
            logger.error(f"Error in semantic sectioning: {e}")
            # Fallback: create simple sections based on text length
            lines = [line for line in enumerated_text.split('\n') if line.strip()]
            section_size = max(10, len(lines) // 3)  
            sections = []
            
            for i in range(0, len(lines), section_size):
                start = i + 1
                end = min(i + section_size, len(lines))
                sections.append({
                    "start_sentence": start,
                    "end_sentence": end,
                    "title": f"Section {len(sections) + 1}"
                })
            
            return sections
    
    def extract_section_content(self, enumerated_text: str, start_sentence: int, end_sentence: int) -> str:
        """Extract content for a specific section based on sentence numbers"""
        lines = enumerated_text.strip().split('\n')
        section_lines = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                # Extract sentence number
                match = re.match(r'<(\d+)>', line)
                if match:
                    sentence_num = int(match.group(1))
                    if start_sentence <= sentence_num <= end_sentence:
                        # Remove the sentence number for the final content
                        content = re.sub(r'^<\d+>\s*', '', line)
                        section_lines.append(content)
        
        return '\n'.join(section_lines)
    
    def get_page_numbers_for_section(self, section_content: str, pages_map: List[Dict], combined_text: str) -> List[int]:
        """
        Determine which pages a section spans based on character positions
        
        Args:
            section_content: The content of the section
            pages_map: Page mapping from PDF extraction
            combined_text: Original combined text from PDF
            
        Returns:
            List of page numbers the section spans
        """
        # Find the start and end positions of the section in the combined text
        section_start = combined_text.find(section_content[:100])  # Use first 100 chars for matching
        if section_start == -1:
            # Fallback: try to match with more flexible approach
            words = section_content.split()[:10]  # First 10 words
            search_text = ' '.join(words)
            section_start = combined_text.find(search_text)
        
        if section_start == -1:
            logger.warning("Could not find section in combined text, assigning to page 1")
            return [1]
        
        section_end = section_start + len(section_content)
        page_numbers = set()
        
        for page_info in pages_map:
            if page_info['chunk_type'] == 'text':
                # Check if section overlaps with this page
                if (section_start < page_info['end_char'] and 
                    section_end > page_info['start_char']):
                    page_numbers.add(page_info['page_number'])
        
        return sorted(list(page_numbers)) if page_numbers else [1]
    
    def generate_contextual_header(self, section_title: str, section_content: str, document_title: str = "") -> str:
        """
        Generate contextual header for AutoContext
        
        Args:
            section_title: Title of the section
            section_content: Content of the section
            document_title: Title of the document
            
        Returns:
            Contextual header string
        """
        
        system_prompt = """You are tasked with creating contextual headers for document chunks to improve retrieval accuracy.

Create a concise contextual header that includes:
1. Document-level context (what the document is about)
2. Section-level context (what this specific section covers)
3. Key topics and themes present in the section

The header should be 2-3 sentences that provide essential context for understanding this content chunk.
Keep it factual and descriptive, avoiding subjective language.

Format: Return only the contextual header text, no additional formatting."""

        user_prompt = f"""Document Title: {document_title}
Section Title: {section_title}

Section Content:
{section_content[:500]}{'...' if len(section_content) > 500 else ''}

Generate a contextual header for this section:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating contextual header: {e}")
            # Fallback contextual header
            if document_title:
                return f"This section from '{document_title}' discusses {section_title.lower()}. The content covers key information related to this topic."
            else:
                return f"This section discusses {section_title.lower()}. The content covers key information related to this topic."
    
    def create_final_chunks(self, section: Dict, section_content: str, page_numbers: List[int], 
                          contextual_header: str, section_index: int) -> List[SemanticChunk]:
        """
        Create final chunks from a section, splitting if necessary
        
        Args:
            section: Section metadata
            section_content: Content of the section
            page_numbers: Page numbers the section spans
            contextual_header: Generated contextual header
            section_index: Index of the section
            
        Returns:
            List of SemanticChunk objects
        """
        chunks = []
        
        # If section is small enough, create single chunk
        if len(section_content) <= self.max_chunk_size:
            chunk = SemanticChunk(
                content=f"{contextual_header}\n\n{section_content}",
                section_title=section['title'],
                page_numbers=page_numbers,
                start_sentence=section['start_sentence'],
                end_sentence=section['end_sentence'],
                chunk_id=f"section_{section_index}_chunk_0",
                contextual_header=contextual_header,
                original_content=section_content
            )
            chunks.append(chunk)
        else:
            # Split large section into smaller chunks
            text_chunks = self.text_splitter.split_text(section_content)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = SemanticChunk(
                    content=f"{contextual_header}\n\n{chunk_text}",
                    section_title=section['title'],
                    page_numbers=page_numbers,  # All chunks inherit section's page numbers
                    start_sentence=section['start_sentence'],
                    end_sentence=section['end_sentence'],
                    chunk_id=f"section_{section_index}_chunk_{i}",
                    contextual_header=contextual_header,
                    original_content=chunk_text
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_document(self, pdf_result: Dict[str, Any], document_title: str = "") -> List[SemanticChunk]:
        """
        Main method to process a document with semantic chunking and AutoContext
        
        Args:
            pdf_result: Result from process_pdf function
            document_title: Title of the document
            
        Returns:
            List of SemanticChunk objects ready for RAG pipeline
        """
        logger.info("Starting semantic chunking process...")
        
        enumerated_text, pages_map = self.chunk_and_enumerate_sentences(pdf_result)
        combined_text = pdf_result["combined_text"]

        sections = self.get_semantic_sections(enumerated_text, document_title)
        
        all_chunks = []
        
        for i, section in enumerate(sections):
            logger.info(f"Processing section {i+1}/{len(sections)}: {section['title']}")
            
            # Extract section content
            section_content = self.extract_section_content(
                enumerated_text, 
                section['start_sentence'], 
                section['end_sentence']
            )
            
            # Get page numbers for this section
            page_numbers = self.get_page_numbers_for_section(
                section_content, pages_map, combined_text
            )
            
            # Generate contextual header
            contextual_header = self.generate_contextual_header(
                section['title'], section_content, document_title
            )
            
            # Create final chunks
            section_chunks = self.create_final_chunks(
                section, section_content, page_numbers, contextual_header, i
            )
            
            all_chunks.extend(section_chunks)
        
        logger.info(f"Created {len(all_chunks)} semantic chunks from {len(sections)} sections")
        return all_chunks
    
    def chunks_to_langchain_documents(self, chunks: List[SemanticChunk]) -> List[Document]:
        """
        Convert SemanticChunk objects to LangChain Document objects
        
        Args:
            chunks: List of SemanticChunk objects
            
        Returns:
            List of LangChain Document objects ready for vector store
        """
        documents = []
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "section_title": chunk.section_title,
                    "page_numbers": chunk.page_numbers,
                    "start_sentence": chunk.start_sentence,
                    "end_sentence": chunk.end_sentence,
                    "contextual_header": chunk.contextual_header,
                    "source": "semantic_chunking"
                }
            )
            documents.append(doc)
        
        return documents
    
    def save_chunks_to_json(self, chunks: List[SemanticChunk], output_path: str):
        """
        Save chunks to a JSON file for inspection
        
        Args:
            chunks: List of SemanticChunk objects
            output_path: Path to save the JSON file
        """
        chunks_data = []
        
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.section_title,
                "page_numbers": chunk.page_numbers,
                "start_sentence": chunk.start_sentence,
                "end_sentence": chunk.end_sentence,
                "contextual_header": chunk.contextual_header,
                "original_content": chunk.original_content,
                "full_content": chunk.content,
                "content_length": len(chunk.content)
            }
            chunks_data.append(chunk_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")



def get_chunks(pdf_path: str, document_title: str = None, max_chunk_size: int = 800) -> Tuple[List[Dict], List[Document]]:
    """
    Get semantically split chunks from a PDF file.
    Can be used in another module to get chunksin List[Dict] format and langchain docs also
    This function processes a PDF file and returns semantically coherent chunks
    with contextual headers for improved RAG retrieval performance.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        document_title (str, optional): Title of the document for better context.
                                      If None, will extract from filename.
        max_chunk_size (int, optional): Maximum size for each chunk in characters.
                                      Defaults to 800.
    
    Returns:
        List[Dict]: List of chunk dictionaries with the following structure:
            {
                "chunk_id": str,
                "content": str,  # Full content with contextual header
                "original_content": str,  # Content without header
                "section_title": str,
                "page_numbers": List[int],
                "start_sentence": int,
                "end_sentence": int,
                "contextual_header": str,
                "content_length": int
            }
        Langchaimn documents in format List[Document]
    """

    if not pdf_path:
        raise ValueError("PDF path cannot be empty")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF, got: {pdf_path.suffix}")

    if document_title is None:
        document_title = pdf_path.stem.replace('_', ' ').replace('-', ' ').title()
    
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"Document title: {document_title}")
        
        chunker = SemanticChunker(max_chunk_size=max_chunk_size)
        
        pdf_result = process_pdf(str(pdf_path))
        logger.info("PDF extraction succesful")
        # Validate PDF processing result
        if not pdf_result or not pdf_result.get("combined_text"):
            raise ValueError("PDF processing failed - no text extracted")
        
        # Process the document with semantic chunking
        chunks = chunker.process_document(pdf_result, document_title=document_title)
        langchain_docs = chunker.chunks_to_langchain_documents(chunks)
        
        # Convert SemanticChunk objects to dictionaries
        chunks_dict = []
        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "original_content": chunk.original_content,
                "section_title": chunk.section_title,
                "page_numbers": chunk.page_numbers,
                "start_sentence": chunk.start_sentence,
                "end_sentence": chunk.end_sentence,
                "contextual_header": chunk.contextual_header,
                "content_length": len(chunk.content),
                "original_content_length": len(chunk.original_content),
                "source_file": str(pdf_path),
                "document_title": document_title
            }
            chunks_dict.append(chunk_dict)
        
        logger.info(f"Successfully created {len(chunks_dict)} semantic chunks from {pdf_path}")
        return chunks_dict, langchain_docs
        
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        raise Exception(f"Failed to process PDF: {e}") from e

# try:
#     chunks, Langchain_docs = get_chunks(
#             "C:/Users/tejup/Downloads/extraction_purpose2.pdf",
#             document_title="Sample Document",
#             max_chunk_size=800
#         )
#     logger.info("success!")

# except:
#     raise ValueError("Failed")

# chunk = chunks[0]
# print(chunk.get("section_title"))

def get_chunks_batch(pdf_paths: List[str], max_chunk_size: int = 800) -> Dict[str, List[Dict]]:
    """
    Process multiple PDF files and return chunks for each.
    
    Args:
        pdf_paths (List[str]): List of PDF file paths to process
        max_chunk_size (int, optional): Maximum chunk size. Defaults to 800.
    
    Returns:
        Dict[str, List[Dict]]: Dictionary with filename as key and chunks as value
        
    Example:
        >>> files = ["doc1.pdf", "doc2.pdf"]
        >>> results = get_chunks_batch(files)
        >>> for filename, chunks in results.items():
        ...     print(f"{filename}: {len(chunks)} chunks")
    """
    results = {}
    failed_files = []
    
    for pdf_path in pdf_paths:
        try:
            chunks = get_chunks(pdf_path, max_chunk_size=max_chunk_size)
            filename = Path(pdf_path).name
            results[filename] = chunks
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            failed_files.append(pdf_path)
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
    
    return results

def main():
    
    chunker = SemanticChunker(max_chunk_size=800)
    
    pdf_result = process_pdf("C:/Users/tejup/Downloads/extraction_purpose2.pdf")

    chunks = chunker.process_document(pdf_result, document_title="Sample Document")
    
    # for future RAG use
    langchain_docs = chunker.chunks_to_langchain_documents(chunks)
    
    chunker.save_chunks_to_json(chunks, "base_pipeline/semantic_chunks.json")
    
    print(f"Created {len(chunks)} semantic chunks")
    print(f"Converted to {len(langchain_docs)} LangChain documents")
    
    return chunks, langchain_docs

if __name__ == "__main__":
    main()