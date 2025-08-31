"""
PDF Extractor for EV Charging Station Data Collection
Handles PDF processing with layout preservation and metadata extraction
"""

import os
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
import PyPDF2
from PIL import Image
import pandas as pd
import re
from datetime import datetime
import hashlib

class PDFExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extracted_data = []
        self.supported_formats = config.get('supported_formats', ['pdf', 'PDF'])
        self.max_file_size_mb = config.get('max_file_size_mb', 50)
        self.layout_preservation = config.get('layout_preservation', True)
        self.extract_images = config.get('extract_images', False)
        self.extract_tables = config.get('extract_tables', True)
        
        # Metadata extraction settings
        self.extract_author = config.get('extract_author', True)
        self.extract_date = config.get('extract_date', True)
        self.extract_source = config.get('extract_source', True)
        self.extract_keywords = config.get('extract_keywords', True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # EV-related keywords for content analysis
        self.ev_keywords = [
            'electric vehicle', 'EV', 'charging station', 'charger', 'battery',
            'plug-in', 'hybrid', 'fast charging', 'level 2', 'level 3',
            'DC fast charging', 'Tesla', 'Supercharger', 'ChargePoint',
            'EVgo', 'Electrify America', 'Blink', 'voltage', 'amperage',
            'kilowatt', 'kWh', 'range', 'miles per charge', 'infrastructure',
            'policy', 'regulation', 'standards', 'safety', 'installation'
        ]
    
    def process_pdf_directory(self, pdf_dir: str) -> List[Dict[str, Any]]:
        """Process all PDF files in a directory"""
        pdf_dir_path = Path(pdf_dir)
        if not pdf_dir_path.exists():
            self.logger.error(f"PDF directory does not exist: {pdf_dir}")
            return []
        
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(pdf_dir_path.glob(f"*.{ext}"))
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing PDF: {pdf_file.name}")
                extracted_content = self.extract_pdf_content(pdf_file)
                if extracted_content:
                    self.extracted_data.append(extracted_content)
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        return self.extracted_data
    
    def extract_pdf_content(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Extract content from a single PDF file"""
        try:
            # Check file size
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                self.logger.warning(f"PDF too large ({file_size_mb:.2f}MB): {pdf_path.name}")
                return None
            
            # Extract metadata first
            metadata = self._extract_metadata(pdf_path)
            
            # Extract text content with layout preservation
            text_content = self._extract_text_with_layout(pdf_path)
            
            # Extract tables if enabled
            tables = []
            if self.extract_tables:
                tables = self._extract_tables(pdf_path)
            
            # Extract images if enabled
            images = []
            if self.extract_images:
                images = self._extract_images(pdf_path)
            
            # Analyze content for EV relevance
            ev_relevance = self._analyze_ev_relevance(text_content)
            
            # Create extracted content object
            extracted_content = {
                'file_name': pdf_path.name,
                'file_path': str(pdf_path),
                'file_size_mb': file_size_mb,
                'metadata': metadata,
                'text_content': text_content,
                'tables': tables,
                'images': images,
                'ev_relevance': ev_relevance,
                'extraction_timestamp': time.time(),
                'extraction_method': 'PyMuPDF + pdfplumber'
            }
            
            return extracted_content
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {pdf_path.name}: {e}")
            return None
    
    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = {
            'file_name': pdf_path.name,
            'file_size_bytes': pdf_path.stat().st_size,
            'file_modified': datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
        }
        
        try:
            # Try PyMuPDF first for metadata
            doc = fitz.open(pdf_path)
            if doc.metadata:
                pdf_metadata = doc.metadata
                
                if self.extract_author and pdf_metadata.get('author'):
                    metadata['author'] = pdf_metadata['author']
                
                if self.extract_date and pdf_metadata.get('modDate'):
                    metadata['modification_date'] = pdf_metadata['modDate']
                
                if self.extract_source and pdf_metadata.get('producer'):
                    metadata['producer'] = pdf_metadata['producer']
                
                if self.extract_source and pdf_metadata.get('creator'):
                    metadata['creator'] = pdf_metadata['creator']
                
                metadata['page_count'] = len(doc)
                metadata['pdf_version'] = doc.version
            
            doc.close()
            
        except Exception as e:
            self.logger.warning(f"Error extracting metadata with PyMuPDF: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    info = reader.metadata
                    
                    if self.extract_author and info.get('/Author'):
                        metadata['author'] = info['/Author']
                    
                    if self.extract_date and info.get('/ModDate'):
                        metadata['modification_date'] = info['/ModDate']
                    
                    metadata['page_count'] = len(reader.pages)
                    
            except Exception as fallback_error:
                self.logger.warning(f"Fallback metadata extraction also failed: {fallback_error}")
        
        return metadata
    
    def _extract_text_with_layout(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text content with layout preservation"""
        text_content = {
            'full_text': '',
            'pages': [],
            'layout_blocks': [],
            'headings': [],
            'paragraphs': []
        }
        
        try:
            # Use PyMuPDF for detailed text extraction
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with position information
                blocks = page.get_text("dict")
                page_text = ""
                page_blocks = []
                
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"]
                                if text.strip():
                                    page_text += text + " "
                                    
                                    # Store block with layout information
                                    block_info = {
                                        'text': text,
                                        'bbox': span["bbox"],
                                        'font': span["font"],
                                        'size': span["size"],
                                        'flags': span["flags"],
                                        'page': page_num + 1
                                    }
                                    page_blocks.append(block_info)
                
                # Clean up text
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                
                text_content['pages'].append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'blocks': page_blocks
                })
                
                text_content['full_text'] += page_text + "\n\n"
                text_content['layout_blocks'].extend(page_blocks)
            
            doc.close()
            
            # Extract headings and paragraphs
            text_content['headings'] = self._extract_headings(text_content['layout_blocks'])
            text_content['paragraphs'] = self._extract_paragraphs(text_content['full_text'])
            
        except Exception as e:
            self.logger.warning(f"Error extracting text with PyMuPDF: {e}")
            
            # Fallback to pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text() or ""
                        text_content['pages'].append({
                            'page_number': page_num + 1,
                            'text': page_text
                        })
                        text_content['full_text'] += page_text + "\n\n"
                        
            except Exception as fallback_error:
                self.logger.error(f"Fallback text extraction also failed: {fallback_error}")
        
        return text_content
    
    def _extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # At least header + one row
                            table_data = {
                                'page_number': page_num + 1,
                                'table_number': table_num + 1,
                                'rows': len(table),
                                'columns': len(table[0]) if table else 0,
                                'data': table,
                                'table_type': self._classify_table(table)
                            }
                            tables.append(table_data)
                            
        except Exception as e:
            self.logger.warning(f"Error extracting tables: {e}")
        
        return tables
    
    def _extract_images(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract images from PDF"""
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = {
                                'page_number': page_num + 1,
                                'image_index': img_index,
                                'width': pix.width,
                                'height': pix.height,
                                'colorspace': pix.colorspace.name,
                                'bbox': img[2]  # Image rectangle on page
                            }
                            images.append(img_data)
                        
                        pix = None  # Free memory
                        
                    except Exception as img_error:
                        self.logger.warning(f"Error processing image {img_index} on page {page_num}: {img_error}")
                        continue
            
            doc.close()
            
        except Exception as e:
            self.logger.warning(f"Error extracting images: {e}")
        
        return images
    
    def _extract_headings(self, layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract potential headings based on font size and formatting"""
        headings = []
        
        # Sort blocks by font size to identify potential headings
        sorted_blocks = sorted(layout_blocks, key=lambda x: x.get('size', 0), reverse=True)
        
        # Consider larger fonts as potential headings
        for block in sorted_blocks[:20]:  # Top 20 largest font blocks
            text = block.get('text', '').strip()
            if text and len(text) < 200:  # Headings are usually short
                heading_info = {
                    'text': text,
                    'font_size': block.get('size', 0),
                    'font': block.get('font', ''),
                    'page': block.get('page', 0),
                    'bbox': block.get('bbox', [])
                }
                headings.append(heading_info)
        
        return headings
    
    def _extract_paragraphs(self, full_text: str) -> List[str]:
        """Extract paragraphs from full text"""
        # Split by double newlines and filter out short segments
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
        return paragraphs
    
    def _classify_table(self, table: List[List[str]]) -> str:
        """Classify table type based on content"""
        if not table or len(table) < 2:
            return "unknown"
        
        # Check if it's a data table
        first_row = table[0]
        if any(keyword.lower() in ' '.join(first_row).lower() for keyword in self.ev_keywords):
            return "ev_data"
        
        # Check if it's a pricing table
        if any('$' in cell or 'price' in cell.lower() for cell in first_row):
            return "pricing"
        
        # Check if it's a specification table
        if any('spec' in cell.lower() or 'parameter' in cell.lower() for cell in first_row):
            return "specifications"
        
        return "general"
    
    def _analyze_ev_relevance(self, text_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content relevance to EV charging"""
        full_text = text_content.get('full_text', '').lower()
        
        # Count EV keyword occurrences
        keyword_counts = {}
        for keyword in self.ev_keywords:
            count = full_text.count(keyword.lower())
            if count > 0:
                keyword_counts[keyword] = count
        
        # Calculate relevance score
        total_keywords = sum(keyword_counts.values())
        text_length = len(full_text)
        relevance_score = (total_keywords / max(text_length, 1)) * 1000  # Normalize per 1000 characters
        
        # Classify relevance level
        if relevance_score > 10:
            relevance_level = "high"
        elif relevance_score > 5:
            relevance_level = "medium"
        elif relevance_score > 1:
            relevance_level = "low"
        else:
            relevance_level = "minimal"
        
        return {
            'relevance_score': relevance_score,
            'relevance_level': relevance_level,
            'keyword_counts': keyword_counts,
            'total_keywords_found': total_keywords,
            'text_length': text_length
        }
    
    def save_extracted_data(self, output_path: str = "data/extracted_pdf_data.json"):
        """Save extracted PDF data to JSON file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.extracted_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Extracted PDF data saved to {output_file}")
            
            # Also save summary as CSV
            csv_path = output_path.replace('.json', '_summary.csv')
            summary_data = []
            
            for item in self.extracted_data:
                summary = {
                    'file_name': item['file_name'],
                    'file_size_mb': item['file_size_mb'],
                    'page_count': item['metadata'].get('page_count', 0),
                    'text_length': len(item['text_content'].get('full_text', '')),
                    'tables_count': len(item['tables']),
                    'images_count': len(item['images']),
                    'relevance_score': item['ev_relevance']['relevance_score'],
                    'relevance_level': item['ev_relevance']['relevance_level'],
                    'total_keywords': item['ev_relevance']['total_keywords_found']
                }
                summary_data.append(summary)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            self.logger.info(f"PDF extraction summary saved to {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving extracted data: {e}")
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the PDF extraction process"""
        total_files = len(self.extracted_data)
        total_pages = sum(item['metadata'].get('page_count', 0) for item in self.extracted_data)
        total_text_length = sum(len(item['text_content'].get('full_text', '')) for item in self.extracted_data)
        total_tables = sum(len(item['tables']) for item in self.extracted_data)
        total_images = sum(len(item['images']) for item in self.extracted_data)
        
        # Relevance distribution
        relevance_distribution = {}
        for item in self.extracted_data:
            level = item['ev_relevance']['relevance_level']
            relevance_distribution[level] = relevance_distribution.get(level, 0) + 1
        
        return {
            'total_pdfs_processed': total_files,
            'total_pages_extracted': total_pages,
            'total_text_length': total_text_length,
            'total_tables_extracted': total_tables,
            'total_images_extracted': total_images,
            'average_pages_per_pdf': total_pages / total_files if total_files > 0 else 0,
            'average_text_length_per_pdf': total_text_length / total_files if total_files > 0 else 0,
            'relevance_distribution': relevance_distribution
        }
