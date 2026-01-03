#!/usr/bin/env python3
"""
Generate test files with current affairs Q&A content for testing multi-file upload.
Creates files in various formats: TXT, CSV, JSON, XML, HTML, MD
Note: PDF, DOCX, PPTX, XLSX require additional libraries and are created separately.
"""

import os
import json
from datetime import datetime

# Current Affairs Q&A content
CURRENT_AFFAIRS_QA = """
CURRENT AFFAIRS QUESTIONS AND ANSWERS - 2026

Q1: What is the capital of Australia?
A1: Canberra is the capital of Australia. Many people mistakenly think it's Sydney or Melbourne.

Q2: Who is the current Secretary-General of the United Nations (as of 2024)?
A2: António Guterres from Portugal has been serving as the Secretary-General of the United Nations since January 2017.

Q3: What is the Paris Agreement about?
A3: The Paris Agreement is an international treaty on climate change adopted in 2015. It aims to limit global warming to well below 2°C above pre-industrial levels.

Q4: Which country has the largest population in the world?
A4: India surpassed China to become the world's most populous country in 2023, with over 1.4 billion people.

Q5: What is cryptocurrency and what is Bitcoin?
A5: Cryptocurrency is a digital or virtual currency that uses cryptography for security. Bitcoin, created in 2009, was the first decentralized cryptocurrency.

Q6: What are the BRICS nations?
A6: BRICS originally consisted of Brazil, Russia, India, China, and South Africa. In 2024, it expanded to include Egypt, Ethiopia, Iran, Saudi Arabia, and the UAE.

Q7: What is Artificial Intelligence (AI)?
A7: AI is the simulation of human intelligence by machines, including learning, reasoning, and self-correction. Examples include ChatGPT, image recognition, and autonomous vehicles.

Q8: What is the World Economic Forum?
A8: The World Economic Forum (WEF) is an international organization headquartered in Geneva, Switzerland. It hosts the annual Davos meeting of world leaders, CEOs, and intellectuals.

Q9: What is the significance of COP climate conferences?
A9: COP (Conference of the Parties) meetings are annual UN climate change conferences where countries negotiate climate action. COP28 was held in Dubai in 2023.

Q10: What is the International Space Station (ISS)?
A10: The ISS is a modular space station in low Earth orbit. It's a collaborative project involving NASA, Roscosmos, JAXA, ESA, and CSA, and has been continuously occupied since 2000.
"""

def create_txt_file():
    """Create a plain text file."""
    with open("current_affairs_qa.txt", "w") as f:
        f.write(CURRENT_AFFAIRS_QA)
    print("Created: current_affairs_qa.txt")

def create_csv_file():
    """Create a CSV file with Q&A."""
    csv_content = """Question,Answer
What is the capital of Australia?,Canberra is the capital of Australia. Many people mistakenly think it's Sydney or Melbourne.
Who is the current Secretary-General of the United Nations?,António Guterres from Portugal has been serving as the Secretary-General since January 2017.
What is the Paris Agreement about?,The Paris Agreement is an international treaty on climate change adopted in 2015 to limit global warming.
Which country has the largest population in the world?,India surpassed China to become the world's most populous country in 2023.
What is cryptocurrency and what is Bitcoin?,Cryptocurrency is digital currency using cryptography. Bitcoin was the first decentralized cryptocurrency.
What are the BRICS nations?,BRICS originally consisted of Brazil Russia India China and South Africa. Expanded in 2024.
What is Artificial Intelligence?,AI is the simulation of human intelligence by machines including learning and reasoning.
What is the World Economic Forum?,The WEF is an international organization headquartered in Geneva hosting the annual Davos meeting.
What is the significance of COP climate conferences?,COP meetings are annual UN climate change conferences where countries negotiate climate action.
What is the International Space Station?,The ISS is a modular space station in low Earth orbit continuously occupied since 2000.
"""
    with open("current_affairs_qa.csv", "w") as f:
        f.write(csv_content)
    print("Created: current_affairs_qa.csv")

def create_json_file():
    """Create a JSON file with Q&A."""
    qa_data = {
        "title": "Current Affairs Questions and Answers 2026",
        "created_at": datetime.now().isoformat(),
        "questions": [
            {"id": 1, "question": "What is the capital of Australia?", "answer": "Canberra is the capital of Australia."},
            {"id": 2, "question": "Who is the current Secretary-General of the United Nations?", "answer": "António Guterres from Portugal."},
            {"id": 3, "question": "What is the Paris Agreement about?", "answer": "An international treaty on climate change adopted in 2015."},
            {"id": 4, "question": "Which country has the largest population?", "answer": "India surpassed China in 2023."},
            {"id": 5, "question": "What is cryptocurrency?", "answer": "Digital currency using cryptography for security."},
            {"id": 6, "question": "What are the BRICS nations?", "answer": "Brazil, Russia, India, China, South Africa, and more since 2024."},
            {"id": 7, "question": "What is AI?", "answer": "Simulation of human intelligence by machines."},
            {"id": 8, "question": "What is the World Economic Forum?", "answer": "International organization hosting the Davos meeting."},
            {"id": 9, "question": "What is COP?", "answer": "Annual UN climate change conferences."},
            {"id": 10, "question": "What is the ISS?", "answer": "Modular space station in low Earth orbit."}
        ]
    }
    with open("current_affairs_qa.json", "w") as f:
        json.dump(qa_data, f, indent=2)
    print("Created: current_affairs_qa.json")

def create_xml_file():
    """Create an XML file with Q&A."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<quiz>
    <title>Current Affairs Questions and Answers 2026</title>
    <question id="1">
        <text>What is the capital of Australia?</text>
        <answer>Canberra is the capital of Australia.</answer>
    </question>
    <question id="2">
        <text>Who is the current Secretary-General of the United Nations?</text>
        <answer>António Guterres from Portugal.</answer>
    </question>
    <question id="3">
        <text>What is the Paris Agreement about?</text>
        <answer>An international treaty on climate change adopted in 2015.</answer>
    </question>
    <question id="4">
        <text>Which country has the largest population?</text>
        <answer>India surpassed China in 2023.</answer>
    </question>
    <question id="5">
        <text>What is cryptocurrency?</text>
        <answer>Digital currency using cryptography for security.</answer>
    </question>
    <question id="6">
        <text>What are the BRICS nations?</text>
        <answer>Brazil, Russia, India, China, South Africa, and more since 2024.</answer>
    </question>
    <question id="7">
        <text>What is Artificial Intelligence?</text>
        <answer>Simulation of human intelligence by machines.</answer>
    </question>
    <question id="8">
        <text>What is the World Economic Forum?</text>
        <answer>International organization hosting the Davos meeting.</answer>
    </question>
    <question id="9">
        <text>What is COP?</text>
        <answer>Annual UN climate change conferences.</answer>
    </question>
    <question id="10">
        <text>What is the ISS?</text>
        <answer>Modular space station in low Earth orbit.</answer>
    </question>
</quiz>
"""
    with open("current_affairs_qa.xml", "w") as f:
        f.write(xml_content)
    print("Created: current_affairs_qa.xml")

def create_html_file():
    """Create an HTML file with Q&A."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Current Affairs Q&A 2026</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        .qa-item { margin-bottom: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px; }
        .question { font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        .answer { color: #34495e; }
    </style>
</head>
<body>
    <h1>Current Affairs Questions and Answers - 2026</h1>
    
    <div class="qa-item">
        <div class="question">Q1: What is the capital of Australia?</div>
        <div class="answer">Canberra is the capital of Australia. Many people mistakenly think it's Sydney or Melbourne.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q2: Who is the current Secretary-General of the United Nations?</div>
        <div class="answer">António Guterres from Portugal has been serving as the Secretary-General since January 2017.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q3: What is the Paris Agreement about?</div>
        <div class="answer">The Paris Agreement is an international treaty on climate change adopted in 2015 to limit global warming.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q4: Which country has the largest population in the world?</div>
        <div class="answer">India surpassed China to become the world's most populous country in 2023.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q5: What is cryptocurrency?</div>
        <div class="answer">Cryptocurrency is a digital currency using cryptography for security. Bitcoin was the first.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q6: What are the BRICS nations?</div>
        <div class="answer">BRICS originally consisted of Brazil, Russia, India, China, and South Africa. Expanded in 2024.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q7: What is Artificial Intelligence (AI)?</div>
        <div class="answer">AI is the simulation of human intelligence by machines, including learning and reasoning.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q8: What is the World Economic Forum?</div>
        <div class="answer">The WEF is an international organization headquartered in Geneva hosting the annual Davos meeting.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q9: What is the significance of COP climate conferences?</div>
        <div class="answer">COP meetings are annual UN climate change conferences where countries negotiate climate action.</div>
    </div>
    
    <div class="qa-item">
        <div class="question">Q10: What is the International Space Station (ISS)?</div>
        <div class="answer">The ISS is a modular space station in low Earth orbit, continuously occupied since 2000.</div>
    </div>
</body>
</html>
"""
    with open("current_affairs_qa.html", "w") as f:
        f.write(html_content)
    print("Created: current_affairs_qa.html")

def create_md_file():
    """Create a Markdown file with Q&A."""
    md_content = """# Current Affairs Questions and Answers - 2026

## General Knowledge Quiz

### Q1: What is the capital of Australia?
**Answer:** Canberra is the capital of Australia. Many people mistakenly think it's Sydney or Melbourne.

### Q2: Who is the current Secretary-General of the United Nations?
**Answer:** António Guterres from Portugal has been serving as the Secretary-General since January 2017.

### Q3: What is the Paris Agreement about?
**Answer:** The Paris Agreement is an international treaty on climate change adopted in 2015 to limit global warming to well below 2°C.

### Q4: Which country has the largest population in the world?
**Answer:** India surpassed China to become the world's most populous country in 2023, with over 1.4 billion people.

### Q5: What is cryptocurrency and what is Bitcoin?
**Answer:** Cryptocurrency is digital or virtual currency using cryptography for security. Bitcoin, created in 2009, was the first decentralized cryptocurrency.

### Q6: What are the BRICS nations?
**Answer:** BRICS originally consisted of Brazil, Russia, India, China, and South Africa. In 2024, it expanded to include Egypt, Ethiopia, Iran, Saudi Arabia, and the UAE.

### Q7: What is Artificial Intelligence (AI)?
**Answer:** AI is the simulation of human intelligence by machines, including learning, reasoning, and self-correction.

### Q8: What is the World Economic Forum?
**Answer:** The World Economic Forum (WEF) is an international organization headquartered in Geneva, Switzerland, hosting the annual Davos meeting.

### Q9: What is the significance of COP climate conferences?
**Answer:** COP (Conference of the Parties) meetings are annual UN climate change conferences where countries negotiate climate action.

### Q10: What is the International Space Station (ISS)?
**Answer:** The ISS is a modular space station in low Earth orbit, a collaborative project involving NASA, Roscosmos, JAXA, ESA, and CSA.

---
*Generated for testing purposes*
"""
    with open("current_affairs_qa.md", "w") as f:
        f.write(md_content)
    print("Created: current_affairs_qa.md")

def main():
    """Generate all test files."""
    print("Generating test files with current affairs Q&A content...\n")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    create_txt_file()
    create_csv_file()
    create_json_file()
    create_xml_file()
    create_html_file()
    create_md_file()
    
    print("\n✅ All text-based test files created!")
    print("\nNote: PDF, DOCX, PPTX, XLSX files need additional libraries.")
    print("You can use the simple text files above for testing, or install:")
    print("  pip install reportlab python-docx python-pptx openpyxl pillow")
    print("Then run: python generate_binary_files.py")

if __name__ == "__main__":
    main()

