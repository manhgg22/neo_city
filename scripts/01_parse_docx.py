from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "raw" / "All database - NEO CITY.docx"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "neo_city_sections.json"

SECTION_TITLES: tuple[tuple[str, str], ...] = (
    ("FACTSHEET", "factsheet"),
    ("LIÊN KẾT VÙNG DỰ ÁN", "location_connectivity"),
    ("Personas TA", "personas"),
    ("Concept", "concept_positioning"),
    ("Định vị", "concept_positioning"),
    ("CHIẾN LƯỢC BÁN HÀNG", "sales_strategy"),
    ("CHÍNH SÁCH BÁN HÀNG", "sales_policy"),
    ("Tình trạng pháp lý", "legal"),
    ("Giá bán & Chính sách bán hàng", "pricing"),
    ("Bản thị trường khu vực Mê Linh", "market"),
    ("Phiếu tính giá theo từng CSBH", "price_sheet"),
)


@dataclass(frozen=True)
class SectionRecord:
    section_key: str
    source_title: str
    raw_text: str


@dataclass
class SectionBuffer:
    section_key: str
    source_title: str
    blocks: list[str] = field(default_factory=list)


def normalized_heading(text: str) -> str:
    """Normalize heading text for resilient matching."""

    stripped = unicodedata.normalize("NFKD", text).strip()
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    without_marks = "".join(
        character for character in stripped if not unicodedata.combining(character)
    )
    return " ".join(without_marks.casefold().split())


SECTION_TITLE_MAP = {
    normalized_heading(title): section_key
    for title, section_key in SECTION_TITLES
}

NON_TITLE_SECTION_STARTS = {
    normalized_heading("CHÍNH SÁCH BÁN HÀNG"),
}


def match_section_key(title: str) -> str | None:
    """Return the target section key for a known major heading."""

    return SECTION_TITLE_MAP.get(normalized_heading(title))


def detect_section_start(title: str, style_name: str) -> str | None:
    """Return a section key when a paragraph marks the start of a major section."""

    normalized_title = normalized_heading(title)
    matched_section_key = SECTION_TITLE_MAP.get(normalized_title)
    if matched_section_key is None:
        return None

    if style_name == "Title" or normalized_title in NON_TITLE_SECTION_STARTS:
        return matched_section_key

    return None


def iter_block_items(parent: DocumentObject | _Cell) -> Iterator[Paragraph | Table]:
    """Yield top-level paragraphs and tables in document order."""

    if isinstance(parent, DocumentObject):
        parent_element = parent.element.body
    elif isinstance(parent, _Cell):
        parent_element = parent._tc
    else:
        raise TypeError(f"Unsupported parent type: {type(parent)!r}")

    for child in parent_element.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


def clean_paragraph_text(text: str) -> str:
    """Normalize paragraph text while preserving readable content."""

    lines = [line.strip() for line in text.replace("\xa0", " ").splitlines()]
    return " ".join(line for line in lines if line)


def escape_markdown_cell(text: str) -> str:
    """Escape cell content for markdown table rendering."""

    return text.replace("|", "\\|")


def extract_cell_text(cell: _Cell) -> str:
    """Return normalized text for a table cell."""

    parts = [clean_paragraph_text(paragraph.text) for paragraph in cell.paragraphs]
    non_empty_parts = [part for part in parts if part]
    return escape_markdown_cell("<br>".join(non_empty_parts))


def render_table_markdown(table: Table) -> str:
    """Render a DOCX table as a readable markdown-style table."""

    rows = [[extract_cell_text(cell) for cell in row.cells] for row in table.rows]
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    separator = ["---"] * width

    markdown_lines = [
        f"| {' | '.join(header)} |",
        f"| {' | '.join(separator)} |",
    ]
    markdown_lines.extend(
        f"| {' | '.join(row)} |" for row in normalized_rows[1:]
    )
    return "\n".join(markdown_lines)


def build_section_record(buffer: SectionBuffer) -> SectionRecord:
    """Finalize a section buffer into an output record."""

    raw_text = "\n\n".join(block for block in buffer.blocks if block.strip()).strip()
    if not raw_text:
        raise ValueError(f"Section '{buffer.source_title}' has empty raw_text.")

    return SectionRecord(
        section_key=buffer.section_key,
        source_title=buffer.source_title,
        raw_text=raw_text,
    )


def extract_sections(document_path: Path) -> list[SectionRecord]:
    """Extract major sections from the source DOCX document."""

    document = Document(document_path)
    sections: list[SectionRecord] = []
    current_section: SectionBuffer | None = None

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            paragraph_text = clean_paragraph_text(block.text)
            if not paragraph_text:
                continue

            matched_section_key = detect_section_start(
                paragraph_text,
                block.style.name,
            )
            if matched_section_key is not None:
                if current_section is not None:
                    sections.append(build_section_record(current_section))

                current_section = SectionBuffer(
                    section_key=matched_section_key,
                    source_title=paragraph_text,
                )
                continue

            if block.style.name == "Title":
                if current_section is not None:
                    sections.append(build_section_record(current_section))
                    current_section = None
                continue

            if current_section is not None:
                current_section.blocks.append(paragraph_text)
            continue

        if current_section is None:
            continue

        table_markdown = render_table_markdown(block)
        if table_markdown:
            current_section.blocks.append(table_markdown)

    if current_section is not None:
        sections.append(build_section_record(current_section))

    return sections


def write_sections_json(sections: list[SectionRecord], output_path: Path) -> None:
    """Write extracted sections to the target JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(section) for section in sections]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_docx_to_json(
    document_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[SectionRecord]:
    """Extract sections from a DOCX document and write them to JSON."""

    sections = extract_sections(document_path)
    write_sections_json(sections, output_path)
    return sections


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Extract major NEO CITY sections from a DOCX document.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the source DOCX file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output JSON file.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for section extraction."""

    args = build_parser().parse_args()
    sections = parse_docx_to_json(args.input, args.output)
    print(f"Extracted {len(sections)} sections to {args.output}")


if __name__ == "__main__":
    main()
