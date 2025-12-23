"""
Wikipedia XML Dump Parser

This script extracts raw text from Wikipedia XML dumps.
Wikipedia dumps can be downloaded from: https://dumps.wikimedia.org/

Example usage:
    # Process in batches of 1000 pages (default, recommended for large dumps)
    python wikipedia_xml_parser.py --input eswiki-latest-pages-articles.xml.bz2 --output articles --batch-mode

    # Process first 5 batches only
    python wikipedia_xml_parser.py --input eswiki-latest-pages-articles.xml.bz2 --output articles --batch-mode --max-batches 5

    # Resume from batch 5 (useful if processing was interrupted)
    python wikipedia_xml_parser.py --input eswiki-latest-pages-articles.xml.bz2 --output articles --batch-mode --start-batch 5

    # Custom batch size of 500 pages
    python wikipedia_xml_parser.py --input eswiki-latest-pages-articles.xml.bz2 --output articles --batch-mode --batch-size 500

    # Single file output (old behavior, not recommended for large dumps)
    python wikipedia_xml_parser.py --input eswiki-latest-pages-articles.xml.bz2 --output articles.txt

    uv run python src/tools/data/wikipedia_xml_parser.py --input data/eswiki-latest-pages-articles-multistream.xml.bz2 --output articles --batch-mode --batch-size 1000
"""

import xml.etree.ElementTree as ET
import argparse
import bz2
import gzip
import re
from typing import Iterator, TextIO, Optional
import sys
from tqdm import tqdm


class WikipediaXMLParser:
    """
    Parser for Wikipedia XML dumps that extracts raw text content.
    """

    def __init__(self, xml_file_path: str):
        """
        Initialize the parser with a Wikipedia XML dump file.

        Args:
            xml_file_path: Path to the XML dump file (.xml, .xml.bz2, or .xml.gz)
        """
        self.xml_file_path = xml_file_path
        self.namespace = '{http://www.mediawiki.org/xml/export-0.10/}'

    def _open_file(self):
        """
        Open the XML file, handling compressed formats.

        Returns:
            File handle for the XML dump
        """
        if self.xml_file_path.endswith('.bz2'):
            return bz2.open(self.xml_file_path, 'rt', encoding='utf-8')
        elif self.xml_file_path.endswith('.gz'):
            return gzip.open(self.xml_file_path, 'rt', encoding='utf-8')
        else:
            return open(self.xml_file_path, 'r', encoding='utf-8')

    def _clean_wikitext(self, text: str) -> str:
        """
        Clean Wikipedia markup from text.

        Args:
            text: Raw wiki markup text

        Returns:
            Cleaned text with markup removed
        """
        # Remove templates ({{...}})
        text = re.sub(r'\{\{[^}]*\}\}', '', text)

        # Remove references (<ref>...</ref>)
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/>', '', text)

        # Remove comments (<!-- ... -->)
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # Remove file/image links
        text = re.sub(r'\[\[(?:File|Image|Archivo|Imagen):[^\]]*\]\]', '', text, flags=re.IGNORECASE)

        # Convert internal links [[link|text]] to just text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

        # Remove external links
        text = re.sub(r'\[http[^\]]*\]', '', text)

        # Remove bold and italic markup
        text = re.sub(r"'''([^']+)'''", r'\1', text)
        text = re.sub(r"''([^']+)''", r'\1', text)

        # Remove headings (== text ==)
        text = re.sub(r'={2,}([^=]+)={2,}', r'\1', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove multiple whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def parse_pages(self, clean_markup: bool = True) -> Iterator[dict]:
        """
        Parse Wikipedia pages from the XML dump.

        Args:
            clean_markup: If True, remove Wikipedia markup from text

        Yields:
            Dictionary containing page information:
                - title: Page title
                - id: Page ID
                - text: Page text content
                - namespace: Page namespace
        """
        with self._open_file() as file:
            # Use iterparse for memory-efficient processing of large files
            context = ET.iterparse(file, events=('start', 'end'))
            context = iter(context)

            # Get root element
            _, root = next(context)

            current_page = {}

            for event, elem in context:
                tag = elem.tag.replace(self.namespace, '')

                if event == 'end':
                    if tag == 'title':
                        current_page['title'] = elem.text or ''
                    elif tag == 'id' and 'id' not in current_page:
                        current_page['id'] = elem.text or ''
                    elif tag == 'ns':
                        current_page['namespace'] = elem.text or '0'
                    elif tag == 'text':
                        text = elem.text or ''
                        if clean_markup:
                            text = self._clean_wikitext(text)
                        current_page['text'] = text
                    elif tag == 'page':
                        # Yield the complete page
                        if current_page.get('text'):
                            yield current_page
                        current_page = {}
                        # Clear the element to free memory
                        elem.clear()
                        root.clear()

    def extract_to_file(
        self,
        output_file: str,
        max_pages: Optional[int] = None,
        namespaces: Optional[list] = None,
        clean_markup: bool = True,
        include_title: bool = True
    ):
        """
        Extract text from XML dump and write to a file.

        Args:
            output_file: Path to output text file
            max_pages: Maximum number of pages to extract (None for all)
            namespaces: List of namespace IDs to include (None for all, [0] for main articles)
            clean_markup: If True, remove Wikipedia markup
            include_title: If True, include page titles in output
        """
        namespaces = namespaces or [0]  # Default to main article namespace
        page_count = 0

        with open(output_file, 'w', encoding='utf-8') as out:
            for page in self.parse_pages(clean_markup=clean_markup):
                # Filter by namespace
                if namespaces and page.get('namespace', '0') not in [str(ns) for ns in namespaces]:
                    continue

                # Write to file
                if include_title:
                    out.write(f"=== {page['title']} ===\n")
                out.write(page['text'])
                out.write('\n\n')

                page_count += 1

                # Progress indicator
                if page_count % 1000 == 0:
                    print(f"Processed {page_count} pages...", file=sys.stderr)

                # Check max pages limit
                if max_pages and page_count >= max_pages:
                    break

        print(f"Extraction complete. Total pages: {page_count}", file=sys.stderr)

    def extract_batches(
        self,
        output_prefix: str,
        batch_size: int = 1000,
        max_batches: Optional[int] = None,
        start_batch: int = 0,
        namespaces: Optional[list] = None,
        clean_markup: bool = True,
        include_title: bool = True
    ):
        """
        Extract text from XML dump in batches, creating separate files for each batch.

        Args:
            output_prefix: Prefix for output files (e.g., 'articles' creates 'articles_batch_0.txt', 'articles_batch_1.txt', etc.)
            batch_size: Number of pages per batch (default: 1000)
            max_batches: Maximum number of batches to process (None for all)
            start_batch: Batch number to start from (default: 0, useful for resuming)
            namespaces: List of namespace IDs to include (None for all, [0] for main articles)
            clean_markup: If True, remove Wikipedia markup
            include_title: If True, include page titles in output
        """
        namespaces = namespaces or [0]  # Default to main article namespace
        page_count = 0
        batch_count = 0
        current_batch_pages = 0
        current_file = None

        try:
            for page in tqdm(self.parse_pages(clean_markup=clean_markup)):
                # Filter by namespace
                if namespaces and page.get('namespace', '0') not in [str(ns) for ns in namespaces]:
                    continue

                # Skip pages until we reach the start batch
                if batch_count < start_batch:
                    page_count += 1
                    if page_count % batch_size == 0:
                        batch_count += 1
                        print(f"Skipping batch {batch_count - 1}...", file=sys.stderr)
                    continue

                # Open new batch file if needed
                if current_file is None or current_batch_pages >= batch_size:
                    if current_file is not None:
                        current_file.close()
                        print(f"Completed batch {batch_count - 1} ({current_batch_pages} pages)", file=sys.stderr)

                    # Check if we've reached max batches
                    if max_batches is not None and batch_count >= start_batch + max_batches:
                        break

                    output_file = f"{output_prefix}_batch_{batch_count}.txt"
                    current_file = open(output_file, 'w', encoding='utf-8')
                    print(f"Starting batch {batch_count}: {output_file}", file=sys.stderr)
                    current_batch_pages = 0
                    batch_count += 1

                # Write to current batch file
                if include_title:
                    current_file.write(f"=== {page['title']} ===\n")
                current_file.write(page['text'])
                current_file.write('\n\n')

                page_count += 1
                current_batch_pages += 1

                # Progress indicator within batch
                if current_batch_pages % 100 == 0:
                    print(f"  Batch {batch_count - 1}: {current_batch_pages}/{batch_size} pages...", file=sys.stderr)

        finally:
            if current_file is not None:
                current_file.close()
                print(f"Completed batch {batch_count - 1} ({current_batch_pages} pages)", file=sys.stderr)

        print(f"\nExtraction complete. Total pages: {page_count}, Batches: {batch_count - start_batch}", file=sys.stderr)


def main():
    """
    Main entry point for command-line usage.
    """
    parser = argparse.ArgumentParser(
        description='Extract raw text from Wikipedia XML dumps'
    )
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Input XML dump file (.xml, .xml.bz2, or .xml.gz)'
    )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help='Output file path (or prefix for batch mode)'
    )
    parser.add_argument(
        '--batch-mode',
        '-b',
        action='store_true',
        help='Enable batch processing (creates multiple output files)'
    )
    parser.add_argument(
        '--batch-size',
        '-s',
        type=int,
        default=1000,
        help='Number of pages per batch (default: 1000, only used with --batch-mode)'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum number of batches to process (default: all, only used with --batch-mode)'
    )
    parser.add_argument(
        '--start-batch',
        type=int,
        default=0,
        help='Batch number to start from (default: 0, useful for resuming, only used with --batch-mode)'
    )
    parser.add_argument(
        '--max-pages',
        '-m',
        type=int,
        default=None,
        help='Maximum number of pages to extract (default: all, only used without --batch-mode)'
    )
    parser.add_argument(
        '--namespaces',
        '-n',
        type=int,
        nargs='+',
        default=[0],
        help='Namespace IDs to include (default: 0 for main articles)'
    )
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Keep Wikipedia markup (default: remove markup)'
    )
    parser.add_argument(
        '--no-titles',
        action='store_true',
        help='Do not include page titles in output'
    )

    args = parser.parse_args()

    print(f"Parsing Wikipedia dump: {args.input}", file=sys.stderr)

    wiki_parser = WikipediaXMLParser(args.input)

    if args.batch_mode:
        print(f"Batch mode enabled: {args.batch_size} pages per batch", file=sys.stderr)
        if args.start_batch > 0:
            print(f"Resuming from batch {args.start_batch}", file=sys.stderr)

        wiki_parser.extract_batches(
            output_prefix=args.output,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            start_batch=args.start_batch,
            namespaces=args.namespaces,
            clean_markup=not args.no_clean,
            include_title=not args.no_titles
        )

        print(f"Batch files saved with prefix: {args.output}", file=sys.stderr)
    else:
        wiki_parser.extract_to_file(
            output_file=args.output,
            max_pages=args.max_pages,
            namespaces=args.namespaces,
            clean_markup=not args.no_clean,
            include_title=not args.no_titles
        )

        print(f"Output saved to: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
