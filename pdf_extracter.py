import os
import pdfplumber
import logging
from typing import List, Dict, Any, Optional
import json
# from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()
# Configure logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Utility: Clean and normalize table cells

def clean_cell(cell) -> str:
    """
    Convert a cell value to a clean string, handling None values and whitespace.
    """
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell.strip()
    return str(cell).strip()

def clean_table(table: List[List[Any]]) -> List[List[str]]:
    """
    Clean all cells in a table and return a normalized table with string values.
    """
    if not table:
        return []
    
    cleaned_table = []
    for row in table:
        if row is None:
            continue
        cleaned_row = [clean_cell(cell) for cell in row]
        cleaned_table.append(cleaned_row)
    
    return cleaned_table


# Utility: Get table dimensions and stats

def get_table_stats(table: List[List[Any]]) -> Dict[str, Any]:
    """
    Analyzes a table and returns useful statistics about it.
    Now handles None values properly.
    """
    if not table:
        return {"col_count": 0, "row_count": 0, "empty": True}
    
    # Clean the table first
    cleaned_table = clean_table(table)
    
    if not cleaned_table:
        return {"col_count": 0, "row_count": 0, "empty": True}
    
    # Filter out completely empty rows
    non_empty_rows = [row for row in cleaned_table if any(cell for cell in row)]
    
    if not non_empty_rows:
        return {"col_count": 0, "row_count": 0, "empty": True}
    
    # Find the maximum column count (some rows might have different lengths)
    col_count = max(len(row) for row in non_empty_rows)
    
    # Get the first non-empty row for header analysis
    first_row = non_empty_rows[0]
    
    # Check if first row looks like a header
    has_header_format = False
    
    # If first row cells are mostly text (no numbers) and other rows have numbers, likely a header
    first_row_text_cells = 0
    for cell in first_row:
        if cell and not any(c.isdigit() for c in cell):
            first_row_text_cells += 1
    
    # Header detection: If >70% of cells are text-only (no digits) it might be a header
    if len(first_row) > 0:
        text_cell_ratio = first_row_text_cells / len(first_row)
        has_header_format = text_cell_ratio > 0.7
    
    return {
        "col_count": col_count,
        "row_count": len(non_empty_rows),
        "empty": False,
        "has_header_format": has_header_format,
        "first_row": first_row,
        "cleaned_table": non_empty_rows
    }


# Utility: Convert table to formatted text with tags

def format_table_with_tags(
    table_rows: List[List[str]],
    start_page: int,
    table_id: str,
) -> str:
    """
    Format table with start/end tags and structured table data.
    No LLM processing - just clean formatting.
    """
    if not table_rows:
        return f"<TABLE_START id='{table_id}' page='{start_page}'>\n[Empty table]\n<TABLE_END id='{table_id}'>"

    # Clean the table rows
    cleaned_rows = clean_table(table_rows)
    
    # Filter out empty rows
    filtered_rows = [row for row in cleaned_rows if any(cell for cell in row)]
    
    if not filtered_rows:
        return f"<TABLE_START id='{table_id}' page='{start_page}'>\n[Empty table]\n<TABLE_END id='{table_id}'>"

    # Create structured table representation
    table_lines = []
    max_cols = max(len(r) for r in filtered_rows) if filtered_rows else 0
    
    for i, row in enumerate(filtered_rows):
        # Pad row to consistent length
        padded_row = row + [""] * (max_cols - len(row))
        
        # Join with pipe separator for readability
        row_text = " | ".join(cell if cell else "" for cell in padded_row)
        table_lines.append(row_text)
        
        # Add separator after first row if it looks like headers
        if i == 0 and len(filtered_rows) > 1:
            first_row_has_text = any(cell and not cell.replace(".", "").replace(",", "").replace("$", "").replace("%", "").isdigit() 
                                   for cell in padded_row if cell)
            if first_row_has_text:
                separator = " | ".join(["-" * max(3, len(cell)) for cell in padded_row])
                table_lines.append(separator)

    formatted_table = "\n".join(table_lines)
    
    return f"<TABLE_START id='{table_id}' page='{start_page}' rows='{len(filtered_rows)}' cols='{max_cols}'>\n{formatted_table}\n<TABLE_END id='{table_id}'>"


# Heuristic: Determine if a table on this page is a continuation of the previous one

def is_continuation_of(prev_table_stats: Optional[Dict[str, Any]], current_table: List[List[Any]]) -> bool:
    """
    Returns True if 'current_table' likely belongs to the same logical table that 
    started on a previous page.
    """
    if not prev_table_stats or prev_table_stats["empty"]:
        return False
    
    current_stats = get_table_stats(current_table)
    
    if current_stats["empty"]:
        return False
    
    # Different column counts usually mean different tables
    if current_stats["col_count"] != prev_table_stats["col_count"]:
        return False
    
    # If the first row of current table has a header format, it's likely a new table
    if current_stats["has_header_format"]:
        return False
    
    # Additional check: if previous table didn't have headers and current one does,
    # it's probably a new table
    if not prev_table_stats.get("has_header_format", False) and current_stats["has_header_format"]:
        return False
    
    # Made it past all checks - likely a continuation
    return True


# Utility: Sort tables by vertical position

def sort_tables_by_position(page, tables):
    """
    Sort tables based on their vertical position on the page.
    """
    if not tables or len(tables) <= 1:
        return tables
    
    try:
        # Extract table positions using pdfplumber's find_tables
        found_tables = page.find_tables()
        if len(found_tables) != len(tables):
            logger.warning(f"Mismatch between extracted tables ({len(tables)}) and found tables ({len(found_tables)})")
            return tables
        
        # Get positions and sort
        table_positions = []
        for i, found_table in enumerate(found_tables):
            try:
                # y1 is the top y-coordinate (higher values = lower on page in PDF coordinates)
                y_pos = found_table.bbox[1] if found_table.bbox else i
                table_positions.append((i, y_pos))
            except (IndexError, AttributeError):
                table_positions.append((i, i))  # Fallback to original order
        
        # Sort by y-position (descending, so top of page comes first)
        sorted_indices = [idx for idx, _ in sorted(table_positions, key=lambda x: -x[1])]
        return [tables[i] for i in sorted_indices]
        
    except Exception as e:
        logger.warning(f"Could not sort tables by position: {e}. Using original order.")
        return tables


# Validate PDF exists and is readable

def validate_pdf(pdf_path: str) -> bool:
    """
    Check if PDF file exists and is readable.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Just check if we can get page count
            page_count = len(pdf.pages)
            if page_count == 0:
                logger.error(f"PDF has no pages: {pdf_path}")
                return False
        return True
    except Exception as e:
        logger.error(f"Cannot open PDF: {pdf_path}. Error: {e}")
        return False


# Main processing: extract text + handle tables (with multi‐page continuity)

def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Process PDF with table tagging (no LLM processing).
    """
    # Validate PDF first
    if not validate_pdf(pdf_path):
        return {
            "combined_text": "",
            "pages_map": [],
            "chunks": [],
            "error": "Invalid or inaccessible PDF file"
        }
    
    combined_chunks: List[Dict[str, Any]] = []
    ongoing_table_rows: List[List[str]] = []
    ongoing_table_start: Optional[int] = None
    ongoing_table_stats: Optional[Dict[str, Any]] = None
    table_counter = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Opened PDF: {pdf_path} ({total_pages} pages)")

            for page_number, page in enumerate(pdf.pages, start=1):
                logger.info(f"Processing page {page_number}/{total_pages}")

                
                # 1. Extract raw text of this page
                
                try:
                    raw_text = page.extract_text() or ""
                    if not raw_text.strip():
                        raw_text = f"[Page {page_number} contains no extractable text]"
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_number}: {e}")
                    raw_text = f"[Error extracting text from page {page_number}: {str(e)}]"

                
                # 2. Extract and sort tables on this page
                
                current_page_tables = []
                try:
                    extracted_tables = page.extract_tables()
                    if extracted_tables:
                        current_page_tables = sort_tables_by_position(page, extracted_tables)
                        logger.info(f"Found {len(current_page_tables)} tables on page {page_number}")
                except Exception as e:
                    logger.error(f"Error extracting tables from page {page_number}: {e}")
                    current_page_tables = []

                
                # 3. Handle table continuation logic
                
                if ongoing_table_rows and current_page_tables:
                    # Check if first table on this page continues the previous table
                    first_table = current_page_tables[0]
                    if not is_continuation_of(ongoing_table_stats, first_table):
                        # Finalize the previous table
                        table_counter += 1
                        logger.info(f"Finalizing table {table_counter} that started on page {ongoing_table_start}")
                        formatted_table = format_table_with_tags(
                            ongoing_table_rows,
                            start_page=ongoing_table_start,
                            table_id=f"table_{table_counter}",
                        )
                        combined_chunks.append({
                            "content": formatted_table,
                            "page_number": ongoing_table_start,
                            "chunk_type": "table",
                            "table_id": f"table_{table_counter}",
                        })
                        # Reset buffer
                        ongoing_table_rows = []
                        ongoing_table_start = None
                        ongoing_table_stats = None

                
                # 4. Process tables on this page
                
                for table_idx, tbl in enumerate(current_page_tables):
                    try:
                        # Get statistics about this table
                        curr_stats = get_table_stats(tbl)
                        
                        if curr_stats["empty"]:
                            logger.info(f"Skipping empty table {table_idx + 1} on page {page_number}")
                            continue

                        if ongoing_table_rows and is_continuation_of(ongoing_table_stats, tbl):
                            # This table is a continuation
                            logger.info(f"Table {table_idx + 1} on page {page_number} continues from page {ongoing_table_start}")
                            cleaned_table = curr_stats["cleaned_table"]
                            ongoing_table_rows.extend(cleaned_table)
                        else:
                            # Finalize any previous ongoing table first
                            if ongoing_table_rows:
                                table_counter += 1
                                logger.info(f"Finalizing table {table_counter} that started on page {ongoing_table_start}")
                                formatted_table = format_table_with_tags(
                                    ongoing_table_rows,
                                    start_page=ongoing_table_start,
                                    table_id=f"table_{table_counter}",
                                )
                                combined_chunks.append({
                                    "content": formatted_table,
                                    "page_number": ongoing_table_start,
                                    "chunk_type": "table",
                                    "table_id": f"table_{table_counter}",
                                })

                            # Start new table
                            logger.info(f"Starting new table {table_idx + 1} on page {page_number}")
                            ongoing_table_rows = curr_stats["cleaned_table"].copy()
                            ongoing_table_start = page_number
                            ongoing_table_stats = curr_stats
                            
                    except Exception as e:
                        logger.error(f"Error processing table {table_idx + 1} on page {page_number}: {e}")
                        continue

                
                # 5. Add page text as a chunk
                
                if raw_text.strip():
                    combined_chunks.append({
                        "content": raw_text.strip(),
                        "page_number": page_number,
                        "chunk_type": "text",
                    })

            
            # 6. Finalize any remaining ongoing table
            
            if ongoing_table_rows:
                table_counter += 1
                logger.info(f"Finalizing final table {table_counter} that started on page {ongoing_table_start}")
                formatted_table = format_table_with_tags(
                    ongoing_table_rows,
                    start_page=ongoing_table_start,
                    table_id=f"table_{table_counter}",
                )
                combined_chunks.append({
                    "content": formatted_table,
                    "page_number": ongoing_table_start,
                    "chunk_type": "table",
                    "table_id": f"table_{table_counter}",
                })

    except Exception as e:
        logger.error(f"Critical error processing PDF: {e}")
        return {
            "combined_text": "",
            "pages_map": [],
            "chunks": [],
            "error": f"Critical error processing PDF: {str(e)}"
        }

    
    # 7. Build combined text and pages map
    
    combined_text = ""
    pages_map: List[Dict[str, Any]] = []

    for chunk in combined_chunks:
        start_idx = len(combined_text)
        content = chunk["content"]
        combined_text += content + "\n\n"
        end_idx = len(combined_text)

        page_entry = {
            "page_number": chunk["page_number"],
            "chunk_type": chunk["chunk_type"],
            "start_char": start_idx,
            "end_char": end_idx,
        }
        
        # Add table_id if it's a table
        if chunk["chunk_type"] == "table":
            page_entry["table_id"] = chunk["table_id"]
            
        pages_map.append(page_entry)

    logger.info(f"Processing complete. Generated {len(combined_chunks)} chunks, {table_counter} tables processed")

    return {
        "combined_text": combined_text,
        "pages_map": pages_map,
        "chunks": combined_chunks,
    }


# Save outputs (combined text + pages map) to disk

def save_outputs(
    combined_text: str,
    pages_map: List[Dict[str, Any]],
    text_path: str = "extracted_full_text.txt",
    map_path: str = "pages_map.json"
) -> None:
    try:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
        logger.info(f"Combined text saved to {text_path}")
    except Exception as e:
        logger.error(f"Failed writing combined text: {e}")

    try:
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(pages_map, f, indent=2, ensure_ascii=False)
        logger.info(f"Pages map saved to {map_path}")
    except Exception as e:
        logger.error(f"Failed writing pages map: {e}")


# Entry point

def main():
    # Get PDF path from environment variable or use default
    pdf_path = os.environ.get("PDF_PATH", "C:/Users/tejup/Downloads/extraction_purpose2.pdf")
    
    print(f"Processing PDF: {pdf_path}")
    result = process_pdf(pdf_path)
    
    # Check if processing was successful
    if "error" in result:
        logger.error(f"PDF processing failed: {result['error']}")
        print(f"ERROR: {result['error']}")
        return
    
    # Save combined extracted text and pages_map
    save_outputs(
        combined_text=result["combined_text"],
        pages_map=result["pages_map"]
    )

    # Print a brief preview
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total characters in combined_text: {len(result['combined_text'])}")
    print(f"Total chunks recorded: {len(result['chunks'])}")
    print(f"Total page‐map entries: {len(result['pages_map'])}")
    
    # Count tables
    table_count = sum(1 for chunk in result['chunks'] if chunk['chunk_type'] == 'table')
    text_count = sum(1 for chunk in result['chunks'] if chunk['chunk_type'] == 'text')
    print(f"Tables extracted: {table_count}")
    print(f"Text chunks extracted: {text_count}")
    
    print("\n=== PREVIEW OF EXTRACTED TEXT ===")
    preview_length = min(500, len(result["combined_text"]))
    print(result["combined_text"][:preview_length] + ("..." if preview_length < len(result["combined_text"]) else ""))

if __name__ == "__main__":
    main()