#!/usr/bin/env python3
"""
Extension support for Probability Stasis
Handles: .csv (original) and .txt (smart parsing) files
"""

import pandas as pd
import re
from pathlib import Path

def load_dataset(filepath="data/chat_messages.csv"):
    """
    Universal loader for CSV and TXT files.
    Returns: (df, text_col, user_col)
    """
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Handle .txt files with smart parsing
    if filepath.endswith('.txt'):
        print(f"📄 Loading TXT with smart parsing: {filepath}")
        
        # Try multiple encodings
        content = None
        for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.read()
                print(f"   (Detected encoding: {enc})")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise UnicodeDecodeError(f"Could not decode {filepath}")
        
        # Parse sections with intelligent extraction
        raw_sections = [s.strip() for s in content.split('\n\n\n') if s.strip()]
        parsed_docs = []
        
        for section in raw_sections:
            # Check if this is a roster table
            if is_roster_table(section):
                # Extract individual people
                people = extract_people_from_roster(section)
                parsed_docs.extend(people)
                print(f"   📋 Parsed {len(people)} people from roster")
            else:
                # Keep non-roster sections as-is (but split by double newline if needed)
                sub_sections = [s.strip() for s in section.split('\n\n') if s.strip()]
                parsed_docs.extend(sub_sections)
        
        print(f"📝 Total parsed documents: {len(parsed_docs)}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'source': ['team_note'] * len(parsed_docs),
            'body_full': parsed_docs
        })
        
        # Show sample
        print(f"\n📋 Sample documents:")
        for i, doc in enumerate(parsed_docs[:3], 1):
            preview = doc.replace('\n', ' ')[:100]
            print(f"  {i}. {preview}...")
        
        return df, 'body_full', 'source'
    
    # Handle .csv files (original behavior)
    else:
        print(f"📊 Loading CSV: {filepath}")
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        
        text_candidates = ["body_full", "body", "text", "message", "content"]
        user_candidates = ["login", "user", "username", "author", "source"]
        
        text_col = next((c for c in text_candidates if c in df.columns), None)
        if text_col is None:
            text_col = max(df.columns, key=lambda c: df[c].astype(str).map(len).mean())
        user_col = next((c for c in user_candidates if c in df.columns), None)
        
        print(f"🧾 Columns: text='{text_col}'" + (f", user='{user_col}'" if user_col else ""))
        df[text_col] = df[text_col].astype(str).fillna("")
        
        return df, text_col, user_col

def is_roster_table(section):
    """Detect if section is a staff roster table"""
    lines = section.split('\n')
    
    # Check for roster indicators
    has_first_name = any('First Name' in line for line in lines[:5])
    has_last_name = any('Last Name' in line for line in lines[:5])
    has_role = any('Role' in line for line in lines[:5])
    
    # Or check for numbered list pattern with names
    numbered_pattern = re.compile(r'^\s*\d+\s*$')
    numbered_lines = [line for line in lines if numbered_pattern.match(line)]
    
    return (has_first_name and has_last_name and has_role) or len(numbered_lines) >= 3

def extract_people_from_roster(section):
    """Extract individual people from roster table format"""
    lines = [line.strip() for line in section.split('\n') if line.strip()]
    people = []
    
    # Find header position
    header_idx = -1
    for i, line in enumerate(lines):
        if 'First Name' in line or 'Role' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        # Try alternative: numbered list format
        return extract_numbered_roster(lines)
    
    # Parse rows after header
    i = header_idx + 1
    while i < len(lines):
        # Look for row number
        if lines[i].isdigit():
            row_num = lines[i]
            # Collect next 3-4 lines as one entry
            entry_lines = [row_num]
            i += 1
            
            # Collect until next number or empty
            while i < len(lines) and not lines[i].isdigit():
                entry_lines.append(lines[i])
                i += 1
            
            # Format as clean document
            if len(entry_lines) >= 3:
                person_doc = format_person_entry(entry_lines)
                if person_doc:
                    people.append(person_doc)
        else:
            i += 1
    
    return people

def extract_numbered_roster(lines):
    """Extract from simple numbered lists"""
    people = []
    current_entry = []
    
    for line in lines:
        if line.isdigit() and current_entry:
            # Save previous entry
            if len(current_entry) >= 2:
                doc = format_person_entry(current_entry)
                if doc:
                    people.append(doc)
            current_entry = [line]
        else:
            current_entry.append(line)
    
    # Don't forget last entry
    if current_entry and len(current_entry) >= 2:
        doc = format_person_entry(current_entry)
        if doc:
            people.append(doc)
    
    return people

def format_person_entry(entry_lines):
    """Format a person entry into a clean text document"""
    # Remove empty lines and row number
    clean_lines = [l for l in entry_lines if l and not l.isdigit()]
    
    if len(clean_lines) < 2:
        return None
    
    # Try to extract structured info
    if len(clean_lines) >= 3:
        # Assume: First Name, Last Name, Role (and possibly more)
        first = clean_lines[0]
        last = clean_lines[1]
        role = clean_lines[2]
        
        # Build clean document
        doc = f"Name: {first} {last}\nRole: {role}"
        
        # Add any additional info
        if len(clean_lines) > 3:
            extra = ' '.join(clean_lines[3:])
            doc += f"\nDetails: {extra}"
        
        return doc
    else:
        # Simple format
        return ' '.join(clean_lines)

