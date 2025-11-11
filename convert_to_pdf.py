#!/usr/bin/env python3
"""
Convert PROJECT_REPORT.md to a well-structured PDF
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path

# Read the markdown file
md_file = Path("PROJECT_REPORT.md")
md_content = md_file.read_text(encoding='utf-8')

# Convert markdown to HTML with extensions for better formatting
md = markdown.Markdown(extensions=[
    'extra',  # Adds support for tables, fenced code blocks, etc.
    'codehilite',  # Syntax highlighting for code blocks
    'toc',  # Table of contents
    'nl2br',  # Newline to break
    'sane_lists'  # Better list handling
])

html_body = md.convert(md_content)

# Create a complete HTML document with styling
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Weld Quality Analysis and Prediction - Project Report</title>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm 2cm 2.5cm 2cm;
            @bottom-center {{
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #666;
            }}
        }}
        
        body {{
            font-family: 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            font-size: 11pt;
            max-width: 100%;
        }}
        
        h1 {{
            color: #1a472a;
            font-size: 24pt;
            margin-top: 0;
            margin-bottom: 10pt;
            border-bottom: 3px solid #2e7d32;
            padding-bottom: 8pt;
            page-break-after: avoid;
        }}
        
        h2 {{
            color: #2e7d32;
            font-size: 18pt;
            margin-top: 24pt;
            margin-bottom: 12pt;
            border-bottom: 2px solid #66bb6a;
            padding-bottom: 6pt;
            page-break-after: avoid;
        }}
        
        h3 {{
            color: #388e3c;
            font-size: 14pt;
            margin-top: 18pt;
            margin-bottom: 10pt;
            page-break-after: avoid;
        }}
        
        h4 {{
            color: #43a047;
            font-size: 12pt;
            margin-top: 14pt;
            margin-bottom: 8pt;
            font-weight: bold;
            page-break-after: avoid;
        }}
        
        p {{
            margin: 8pt 0;
            text-align: justify;
        }}
        
        strong {{
            color: #1b5e20;
            font-weight: bold;
        }}
        
        em {{
            font-style: italic;
            color: #2e7d32;
        }}
        
        ul, ol {{
            margin: 8pt 0;
            padding-left: 25pt;
        }}
        
        li {{
            margin: 4pt 0;
        }}
        
        code {{
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 2pt 4pt;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            color: #c7254e;
        }}
        
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 12pt;
            overflow-x: auto;
            margin: 12pt 0;
            page-break-inside: avoid;
        }}
        
        pre code {{
            background-color: transparent;
            border: none;
            padding: 0;
            color: #333;
            font-size: 9pt;
            line-height: 1.4;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12pt 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }}
        
        th {{
            background-color: #2e7d32;
            color: white;
            font-weight: bold;
            padding: 8pt;
            text-align: left;
            border: 1px solid #1b5e20;
        }}
        
        td {{
            padding: 6pt 8pt;
            border: 1px solid #ddd;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 20pt 0;
        }}
        
        blockquote {{
            border-left: 4px solid #66bb6a;
            padding-left: 12pt;
            margin: 12pt 0;
            color: #555;
            font-style: italic;
            background-color: #f9f9f9;
            padding: 10pt 10pt 10pt 12pt;
        }}
        
        .page-break {{
            page-break-before: always;
        }}
        
        /* Avoid breaking after headers */
        h1, h2, h3, h4, h5, h6 {{
            page-break-after: avoid;
        }}
        
        /* Keep lists together when possible */
        ul, ol {{
            page-break-inside: avoid;
        }}
        
        /* Special styling for checkmarks and crosses */
        p:has(✓) {{
            color: #2e7d32;
        }}
        
        p:has(✗) {{
            color: #c62828;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Convert HTML to PDF
output_file = Path("PROJECT_REPORT.pdf")
print(f"Converting {md_file} to {output_file}...")

HTML(string=html_template).write_pdf(
    output_file,
    stylesheets=[CSS(string="""
        @page {
            size: A4;
            margin: 2.5cm 2cm 2.5cm 2cm;
        }
    """)]
)

print(f"✓ PDF successfully created: {output_file}")
print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

